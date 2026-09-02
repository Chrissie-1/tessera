"""Phase 4: paged attention as a Triton kernel.

`PagedKVCache.gather` rebuilds a contiguous key/value tensor for a sequence on
every decode step, by concatenating its blocks. That gather is pure overhead:
it copies the entire cached context, once per token, only so the model's
attention can read a shape it recognises. It grows with context length, which
is precisely the direction serving pushes.

The kernel here removes it. It walks the block table inside the attention loop
and reads each block straight from the pool, so a sequence's keys and values
are never materialised contiguously at all. Scores are accumulated with an
online softmax, so one pass over the blocks suffices and nothing the size of
the context is ever held in registers or scratch.

Both implementations handle grouped-query attention, where the pool holds
fewer KV heads than the query carries. Each group of query heads reads its
shared KV head in place rather than through an expanded copy of the keys --
expanding them would reintroduce, per step, exactly the copy paging removes.

Two implementations live here on purpose. `paged_attention_torch` is the
readable definition of what paged attention computes, and is what the tests
check the kernel against; it gathers, because being obviously correct is its
only job. `paged_attention_triton` is the fast path. Neither is allowed to
disagree with the other.
"""

from __future__ import annotations

import math

import torch

try:  # pragma: no cover - import guard, exercised by absence not by tests
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def kv_group_size(num_query_heads: int, num_kv_heads: int) -> int:
    """Query heads per KV head, rejecting layouts that do not divide evenly.

    Multi-head attention is the degenerate case where this is one. Anything
    that is not a whole multiple is not grouped-query attention at all, and
    guessing a mapping for it would silently attend to the wrong keys.
    """
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ValueError(
            f"{num_query_heads} query heads do not group evenly "
            f"onto {num_kv_heads} key/value heads"
        )
    return num_query_heads // num_kv_heads


def paged_attention_torch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention over a block table, written for clarity rather than speed.

    Args:
        query: (num_query_heads, head_dim) query for the position being decoded.
        key_cache: (num_blocks, num_kv_heads, block_size, head_dim) pool.
        value_cache: same shape as `key_cache`.
        block_table: (num_used_blocks,) physical block ids, in logical order.
        seq_len: how many cached positions are real; the tail of the last
            block is allocated but not yet written.
        scale: softmax scale, defaulting to 1/sqrt(head_dim).

    Returns:
        (num_query_heads, head_dim) attention output.

    `num_kv_heads` may be smaller than `num_query_heads`: under grouped-query
    attention a group of consecutive query heads shares one KV head. The group
    is folded into the einsum rather than materialised with `repeat_kv`,
    because expanding the keys is the copy paging exists to avoid.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    group = kv_group_size(num_query_heads, num_kv_heads)
    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

    # (num_used_blocks, num_kv_heads, block_size, dim) -> (num_kv_heads, T, dim)
    keys = (
        key_cache[block_table].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
    )
    values = (
        value_cache[block_table].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
    )
    keys = keys[:, :seq_len, :]
    values = values[:, :seq_len, :]

    # Query head q reads KV head q // group, which is the same mapping
    # transformers' `repeat_kv` produces by expanding each KV head in place.
    grouped = query.float().reshape(num_kv_heads, group, head_dim)
    scores = torch.einsum("hgd,htd->hgt", grouped, keys.float()) * scale
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("hgt,htd->hgd", weights, values.float())
    return output.reshape(num_query_heads, head_dim).to(query.dtype)


if HAS_TRITON:  # pragma: no cover - requires a GPU to execute

    @triton.jit
    def _paged_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        block_table_ptr,
        seq_len,
        scale,
        stride_qh,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_oh,
        # Upper-case names are Triton's convention for compile-time
        # constants, so the lint rule does not apply here.
        HEAD_DIM: tl.constexpr,  # noqa: N803
        BLOCK_SIZE: tl.constexpr,  # noqa: N803
        GROUP: tl.constexpr,  # noqa: N803
    ):
        """One program per query head, streaming the block table.

        The running maximum and denominator are carried across blocks so the
        softmax never needs a second pass, which is what lets the kernel touch
        each block exactly once regardless of how long the context is.

        Under grouped-query attention `GROUP` query heads share one KV head,
        so the program's KV head is its own index divided down. Each program
        still reads the pool directly; nothing is expanded to make the head
        counts match.
        """
        head = tl.program_id(0)
        kv_head = head // GROUP

        offs_d = tl.arange(0, HEAD_DIM)
        offs_t = tl.arange(0, BLOCK_SIZE)

        query = tl.load(query_ptr + head * stride_qh + offs_d).to(tl.float32)

        running_max = float("-inf")
        running_sum = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
        for block in range(num_blocks):
            physical = tl.load(block_table_ptr + block)
            positions = block * BLOCK_SIZE + offs_t
            live = positions < seq_len

            key_offsets = (
                physical * stride_kb
                + kv_head * stride_kh
                + offs_t[:, None] * stride_kt
                + offs_d[None, :]
            )
            keys = tl.load(key_ptr + key_offsets, mask=live[:, None], other=0.0)
            keys = keys.to(tl.float32)

            scores = tl.sum(keys * query[None, :], axis=1) * scale
            # Padding at the tail of the final block must not win the softmax.
            scores = tl.where(live, scores, float("-inf"))

            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            rescale = tl.exp(running_max - new_max)
            weights = tl.exp(scores - new_max)
            weights = tl.where(live, weights, 0.0)

            value_offsets = (
                physical * stride_vb
                + kv_head * stride_vh
                + offs_t[:, None] * stride_vt
                + offs_d[None, :]
            )
            values = tl.load(value_ptr + value_offsets, mask=live[:, None], other=0.0)
            values = values.to(tl.float32)

            acc = acc * rescale + tl.sum(weights[:, None] * values, axis=0)
            running_sum = running_sum * rescale + tl.sum(weights, axis=0)
            running_max = new_max

        tl.store(output_ptr + head * stride_oh + offs_d, acc / running_sum)

    def paged_attention_triton(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Kernel-backed paged attention. Same contract as the torch version.

        The block table is never dereferenced on the host: it is passed to the
        kernel and walked there, which is the entire point of the exercise.
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if not query.is_cuda:
            raise ValueError("paged_attention_triton requires CUDA tensors")

        num_query_heads, head_dim = query.shape
        block_size = key_cache.shape[2]
        group = kv_group_size(num_query_heads, key_cache.shape[1])
        scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)

        query = query.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        block_table = block_table.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        output = torch.empty_like(query, dtype=torch.float32)

        _paged_attention_kernel[(num_query_heads,)](
            query,
            key_cache,
            value_cache,
            output,
            block_table,
            seq_len,
            scale,
            query.stride(0),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            output.stride(0),
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            GROUP=group,
        )
        return output.to(query.dtype)

else:  # pragma: no cover - no Triton installed

    def paged_attention_triton(*_args, **_kwargs):
        raise RuntimeError("Triton is not installed")


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Dispatch to the kernel on CUDA, to torch everywhere else.

    Kept as the single entry point so callers never have to ask which
    implementation they are on; the two are tested to agree.
    """
    if HAS_TRITON and query.is_cuda:
        return paged_attention_triton(
            query, key_cache, value_cache, block_table, seq_len, scale
        )
    return paged_attention_torch(
        query, key_cache, value_cache, block_table, seq_len, scale
    )
