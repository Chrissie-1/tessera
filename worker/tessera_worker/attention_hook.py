"""Phase 4: routing the model's attention through the paged kernel.

`attention.py` computes attention over a block table, but nothing called it.
`PagedEngine` gathered a sequence's blocks back into contiguous tensors on
every decode step and let the model attend to those -- which is precisely the
copy the kernel exists to remove. This module is the missing wire.

transformers 5.x resolves attention through `ALL_ATTENTION_FUNCTIONS`, a
registry a model selects from by name, so no monkeypatching and no forked
modelling code is needed: register a function, name it on the config, and every
layer calls it. The function writes the incoming key and value straight into
the block pool and then calls `paged_attention` -- the dispatcher, so a CUDA
box gets the Triton kernel and a CPU box gets the torch implementation that the
equivalence tests can actually run.

What it deliberately does not do:

* Prefill. `paged_attention` answers for one query position; a prompt is many
  positions, each needing its own causal cut of the context. Prefill stays on
  the model's own attention, where it is already a single dense pass and is
  not the per-token cost the kernel was written to remove.
* Batched decode. `ContinuousBatcher` attends to a whole batch through one
  padded cache; walking a separate block table per row is a different kernel.
* Grouped-query attention. The pool stores KV heads, and folding several query
  heads onto one of them is not something `paged_attention` expresses.

Every one of those falls through to `sdpa_attention_forward`, the exact
function the model was already dispatching to, so an unsupported case is
identical to what it was before the hook existed. For the same reason the hook
declines to install itself unless the model is on `sdpa`: it has one fallback,
and it will not silently substitute it for a different implementation.

Integrating through the dispatcher rather than through
`paged_attention_triton` is what makes this testable: the same wiring runs the
torch implementation on a CPU box, where `test_attention_hook.py` holds it to
the reference engine. On CUDA the identical wiring reaches the Triton kernel,
and that combination has not been executed -- the kernel is tested only against
the torch implementation, in isolation, by the GPU-marked tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from .attention import paged_attention
from .paged import PagedKVCache

try:  # pragma: no cover - import guard, exercised by absence not by tests
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    HAS_ATTENTION_INTERFACE = True
except ImportError:  # pragma: no cover
    sdpa_attention_forward = None
    ALL_ATTENTION_FUNCTIONS = None
    HAS_ATTENTION_INTERFACE = False

logger = logging.getLogger(__name__)

# The name the model's config carries to select this function. Namespaced so
# it cannot collide with an implementation transformers ships later.
PAGED_ATTENTION_IMPLEMENTATION = "tessera_paged"

# The only implementation the hook is willing to displace; see module docstring.
SUPPORTED_BASE_IMPLEMENTATION = "sdpa"


@dataclass(frozen=True)
class PagedDecodeContext:
    """Everything the attention function needs that the model cannot pass it.

    The attention interface hands the function a module, a query and a key --
    it has no channel for "which sequence is this, and where in the pool does
    it live". The engine puts that here for the duration of one forward pass.
    """

    cache: PagedKVCache
    seq_id: int
    position: int
    block_table: torch.Tensor


# A ContextVar rather than a module global: the worker serves requests from a
# thread pool, and two sequences decoding at once must not see each other's
# block table.
_active: ContextVar[PagedDecodeContext | None] = ContextVar(
    "tessera_paged_decode_context", default=None
)


def active_context() -> PagedDecodeContext | None:
    """The context the current forward pass is decoding under, if any."""
    return _active.get()


@contextmanager
def paged_decode(
    cache: PagedKVCache, seq_id: int, position: int
) -> Iterator[PagedDecodeContext]:
    """Route the next forward pass's attention through the block table.

    `position` must already be reserved: the attention function writes this
    step's keys and values into it and then attends over positions
    `[0, position]` inclusive.
    """
    context = PagedDecodeContext(
        cache=cache,
        seq_id=seq_id,
        position=position,
        # Snapshotting the table is safe because `reserve` has already run, so
        # it cannot grow again before the pass ends.
        block_table=cache.block_table_tensor(seq_id),
    )
    token = _active.set(context)
    try:
        yield context
    finally:
        _active.reset(token)


def paged_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
):
    """Attention over the block pool, in the shape the attention interface wants.

    Args:
        module: the attention layer; only `layer_idx` is read from it.
        query: (batch, num_heads, q_len, head_dim).
        key: this step's keys, same layout, not yet in the pool.
        value: this step's values.
        attention_mask: ignored on the paged path -- a single query position
            sits after every cached position, so the causal mask admits all of
            them and there is nothing left to mask.
        scaling: softmax scale chosen by the layer. Passed through rather than
            recomputed, because GPT-2 folds its inverse-layer-index scaling
            into it and 1/sqrt(head_dim) would silently drop that.

    Returns:
        ((batch, q_len, num_heads, head_dim), None) -- head-transposed, which
        is what the caller reshapes into hidden states, and no attention
        weights, which the online-softmax kernel never forms.
    """
    context = _active.get()
    if context is None or not _is_paged_decode(query, key):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    layer = module.layer_idx
    # The current token's keys and values belong in the pool before it is read:
    # a token attends to itself. This is the paging equivalent of the cache
    # update the model would otherwise have done for us.
    context.cache.write(
        context.seq_id, layer, context.position, key[0, :, 0, :], value[0, :, 0, :]
    )
    output = paged_attention(
        query[0, :, 0, :],
        context.cache.layer_keys(layer),
        context.cache.layer_values(layer),
        context.block_table,
        context.position + 1,
        scaling,
    )
    return output.reshape(1, 1, *output.shape), None


def _is_paged_decode(query: torch.Tensor, key: torch.Tensor) -> bool:
    """Whether this call is the one shape the block-table walk handles.

    One sequence, one new position, and one key head per query head. Anything
    else -- prefill, a batched step, grouped-query attention -- is a different
    kernel, so it goes back to the model's own attention.
    """
    batch, num_heads, q_len, _ = query.shape
    return batch == 1 and q_len == 1 and key.shape[1] == num_heads


def enable_paged_attention(model) -> bool:
    """Point `model`'s attention at the block pool. Returns whether it took.

    False is a normal outcome, not an error: a model on an implementation this
    hook cannot fall back to, or a transformers without the attention
    interface, keeps the gather path and stays correct, just slower.
    """
    if not HAS_ATTENTION_INTERFACE:  # pragma: no cover - version-dependent
        logger.warning(
            "transformers has no attention interface; "
            "paged attention stays on the gather path"
        )
        return False

    if model.config._attn_implementation == PAGED_ATTENTION_IMPLEMENTATION:
        return True

    if model.config._attn_implementation != SUPPORTED_BASE_IMPLEMENTATION:
        logger.info(
            "model is on attn_implementation=%s, not %s; "
            "paged attention stays on the gather path",
            model.config._attn_implementation,
            SUPPORTED_BASE_IMPLEMENTATION,
        )
        return False

    ALL_ATTENTION_FUNCTIONS.register(
        PAGED_ATTENTION_IMPLEMENTATION, paged_attention_forward
    )
    model.set_attn_implementation(PAGED_ATTENTION_IMPLEMENTATION)
    # `set_attn_implementation` warns and leaves the config alone for models
    # that do not route through the interface, so confirm rather than assume.
    if model.config._attn_implementation != PAGED_ATTENTION_IMPLEMENTATION:
        logger.warning(
            "%s did not accept a custom attention implementation; "
            "paged attention stays on the gather path",
            type(model).__name__,
        )
        return False

    logger.info("paged attention wired into the decode path")
    return True
