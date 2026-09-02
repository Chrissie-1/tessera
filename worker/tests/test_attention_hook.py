"""Phase 4: paged attention on the decode path.

`test_attention.py` proves the kernel computes the right thing in isolation.
This file proves the decode path actually calls it, which is a separate claim
and the one that was previously false: the kernel existed and nothing reached
it.

So the tests here are deliberately written to fail if the engine quietly went
back to gathering. The dispatcher is counted, not merely allowed to run, and
`PagedKVCache.gather` is asserted *not* to be called during decode -- a test
that passed either way would tell us nothing. Equivalence with the reference
engine is then checked on top, because a kernel that is genuinely on the path
and wrong is worse than one that is not on the path at all.

Everything here runs on CPU, which is the point of integrating via the
dispatcher rather than the Triton function: the torch implementation carries
the same integration logic, so a machine with no GPU can still check it.
"""

from __future__ import annotations

import math

import pytest
import torch

from tessera_worker import attention_hook
from tessera_worker.attention_hook import (
    PAGED_ATTENTION_IMPLEMENTATION,
    paged_attention_forward,
    paged_decode,
)
from tessera_worker.config import num_layers
from tessera_worker.paged import PagedKVCache
from tessera_worker.sampling import SamplingParams


@pytest.fixture(scope="module")
def paged(config):
    from tessera_worker.paged_engine import PagedEngine

    # Block size 4 against prompts of a dozen-odd tokens, so decode crosses
    # block boundaries and the final block is always partly empty.
    return PagedEngine(config, num_blocks=64, block_size=4)


class DispatcherSpy:
    """Counts calls to `paged_attention` and records the context lengths."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls = 0
        self.seq_lens: list[int] = []

    def __call__(self, query, key_cache, value_cache, block_table, seq_len, scale=None):
        self.calls += 1
        self.seq_lens.append(seq_len)
        return self.wrapped(query, key_cache, value_cache, block_table, seq_len, scale)


@pytest.fixture
def spy(monkeypatch):
    spy = DispatcherSpy(attention_hook.paged_attention)
    monkeypatch.setattr(attention_hook, "paged_attention", spy)
    return spy


def decode_steps(result) -> int:
    """How many single-token forward passes a completion required.

    The first token comes out of prefill, and the loop stops before decoding
    past the last one, so a length-capped completion of n tokens takes n-1
    decode steps. An EOS stop takes one more, because the step that produced
    the EOS still ran.
    """
    return result.completion_tokens - (1 if result.finish_reason == "length" else 0)


# -- the hook is installed --------------------------------------------------


def test_paged_attention_is_selected_on_the_model(paged):
    assert paged.paged_attention_enabled
    assert paged.model.config._attn_implementation == PAGED_ATTENTION_IMPLEMENTATION


# -- the decode path genuinely reaches the dispatcher -----------------------


def test_every_decode_step_calls_the_dispatcher_once_per_layer(paged, spy):
    result = paged.generate("The capital of France is", max_tokens=8)

    layers = num_layers(paged.model.config)
    assert decode_steps(result) > 0
    assert spy.calls == layers * decode_steps(result)


def test_decode_does_not_gather(paged, monkeypatch):
    """The gather is the cost paged attention exists to remove.

    Counting dispatcher calls proves the kernel ran; this proves the old path
    did not run beside it. Together they rule out a silent fallback.
    """
    gathers = []
    original = PagedKVCache.gather
    monkeypatch.setattr(
        PagedKVCache,
        "gather",
        lambda self, seq_id: (gathers.append(seq_id), original(self, seq_id))[1],
    )

    result = paged.generate("The capital of France is", max_tokens=8)

    assert decode_steps(result) > 0
    assert gathers == []


def test_context_length_grows_by_one_token_per_step(paged, spy):
    """Each step must attend to exactly one more position than the last.

    A block table read at the wrong length is the failure mode that would
    otherwise pass every shape check and quietly drop or duplicate a token's
    keys.
    """
    prompt = "The capital of France is"
    result = paged.generate(prompt, max_tokens=8)

    layers = num_layers(paged.model.config)
    prompt_tokens = result.prompt_tokens
    expected = [
        prompt_tokens + 1 + step
        for step in range(decode_steps(result))
        for _ in range(layers)
    ]
    assert spy.seq_lens == expected


def test_prefill_deliberately_stays_off_the_kernel(paged, spy):
    """Prefill is many query positions; the kernel answers for one.

    A single-token completion is prefill plus nothing, so the dispatcher must
    not be reached at all. This is a design decision, not an accident: prefill
    is one dense pass either way, and is not the per-token cost being removed.
    """
    paged.generate("The capital of France is", max_tokens=1)

    assert spy.calls == 0


# -- output is still the reference engine's ---------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "The capital of France is",
        "",
        "Numbers: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
    ],
    ids=["short", "empty", "spans_blocks"],
)
def test_paged_attention_greedy_matches_reference(engine, paged, prompt):
    reference = engine.generate(prompt, max_tokens=12)
    actual = paged.generate(prompt, max_tokens=12)

    assert actual.token_ids == reference.token_ids
    assert actual.finish_reason == reference.finish_reason


def test_paged_attention_seeded_sampling_matches_reference(engine, paged):
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=1234)

    reference = engine.generate("Once upon a time", max_tokens=10, params=params)
    actual = paged.generate("Once upon a time", max_tokens=10, params=params)

    assert actual.token_ids == reference.token_ids


def paired_step_logits(engine, paged, prompt: str, steps: int):
    """Decode greedily on the paged engine, pairing each step against dense.

    Comparing tokens is the contract, but on the tiny test model it is a blunt
    instrument: two heads of one dimension each leave the argmax dominated by
    the embeddings, so a wrong context length can survive it. Comparing the
    logits themselves is what actually pins the block-table read.
    """
    prompt_ids = paged.tokenizer.encode(prompt)
    seq_id = paged._new_sequence_id()
    paged.cache.add_sequence(seq_id)
    pairs = []
    try:
        with torch.inference_mode():
            outputs = paged.model(
                input_ids=torch.tensor([prompt_ids], dtype=torch.long),
                use_cache=True,
            )
            paged.cache.append(seq_id, paged.new_kv(outputs.past_key_values, 0))
            tokens = list(prompt_ids)
            logits = outputs.logits[0, -1, :]
            for _ in range(steps):
                tokens.append(int(torch.argmax(logits)))
                logits = paged.decode_step(seq_id, tokens[-1])
                pairs.append((logits, engine.forward_logits(tokens)))
    finally:
        paged.cache.free_sequence(seq_id)
    return pairs


def test_decode_logits_match_a_dense_forward_pass(engine, paged):
    for actual, expected in paired_step_logits(
        engine, paged, "The capital of France is", 8
    ):
        assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("block_size", [1, 3, 8])
def test_matches_reference_when_the_context_straddles_blocks(
    engine, config, block_size
):
    """Cached lengths that are not a multiple of the block size.

    The tail of the final block is allocated but unwritten, and reading it as
    though it held real keys is exactly the bug the online softmax's masking
    is there to prevent. Odd block sizes guarantee we land mid-block, and a
    block size of one guarantees every step allocates.
    """
    from tessera_worker.paged_engine import PagedEngine

    paged = PagedEngine(config, num_blocks=128, block_size=block_size)
    assert paged.paged_attention_enabled

    prompt = "The capital of France is"
    for actual, expected in paired_step_logits(engine, paged, prompt, 9):
        assert torch.allclose(actual, expected, atol=1e-5)

    assert (
        paged.generate(prompt, max_tokens=9).token_ids
        == engine.generate(prompt, max_tokens=9).token_ids
    )


# -- degrading to the gather path -------------------------------------------


def test_engine_falls_back_to_gathering_when_the_hook_cannot_install(
    engine, config, monkeypatch, spy
):
    """An uninstallable hook must cost speed, never correctness."""
    import tessera_worker.paged_engine as paged_engine

    monkeypatch.setattr(paged_engine, "enable_paged_attention", lambda _model: False)
    fallback = paged_engine.PagedEngine(config, num_blocks=64, block_size=4)

    assert not fallback.paged_attention_enabled

    reference = engine.generate("The capital of France is", max_tokens=8)
    actual = fallback.generate("The capital of France is", max_tokens=8)

    assert actual.token_ids == reference.token_ids
    # The counter that proves the kernel runs must also prove it did not.
    assert spy.calls == 0


def test_enable_reports_false_for_an_unsupported_base_implementation():
    """The hook has exactly one fallback, so it will not displace anything else.

    Returning False here is how the engine learns to keep gathering, which is
    the difference between degrading and crashing.
    """

    class Config:
        _attn_implementation = "flex_attention"

    class Model:
        config = Config()

    assert attention_hook.enable_paged_attention(Model()) is False


# -- the attention function in isolation ------------------------------------


def kv_sequence(num_heads: int, tokens: int, head_dim: int):
    torch.manual_seed(0)
    keys = torch.randn(1, num_heads, tokens, head_dim)
    values = torch.randn(1, num_heads, tokens, head_dim)
    return keys, values


class FakeAttention:
    """Stands in for a model's attention layer; only `layer_idx` is read."""

    def __init__(self, layer_idx: int = 0) -> None:
        self.layer_idx = layer_idx


def check_hook_against_dense(cache: PagedKVCache, cached: int, decoys: int = 0):
    """Decode one step through the hook and hold it to textbook attention.

    This is the sharp end of the file. The end-to-end comparisons run on the
    tiny test model, whose two one-dimensional heads make its logits nearly
    indifferent to what attention returns; random keys and values here do not
    let a wrong context length hide.

    `decoys` blocks are handed to another sequence first, so the block table
    under test is not the identity and the indirection has to be real.
    """
    num_heads, head_dim = 2, 4
    if decoys:
        cache.add_sequence(1)
        cache.reserve(1, decoys * cache.block_size)
    cache.add_sequence(0)

    keys, values = kv_sequence(num_heads, cached, head_dim)
    cache.append(0, [(keys, values)])

    new_key = torch.randn(1, num_heads, 1, head_dim)
    new_value = torch.randn(1, num_heads, 1, head_dim)
    query = torch.randn(1, num_heads, 1, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    position = cache.reserve(0, 1)
    assert position == cached
    with paged_decode(cache, 0, position):
        output, weights = paged_attention_forward(
            FakeAttention(), query, new_key, new_value, None, scaling=scale
        )

    full_keys = torch.cat([keys, new_key], dim=2)[0]
    full_values = torch.cat([values, new_value], dim=2)[0]
    scores = torch.einsum("hd,htd->ht", query[0, :, 0, :], full_keys) * scale
    expected = torch.einsum("ht,htd->hd", torch.softmax(scores, dim=-1), full_values)

    assert weights is None
    assert output.shape == (1, 1, num_heads, head_dim)
    assert torch.allclose(output[0, 0], expected, atol=1e-6)


@pytest.mark.parametrize(
    ("block_size", "cached"),
    [(4, 5), (4, 4), (4, 7), (4, 1), (1, 3), (8, 9)],
    ids=[
        "mid_block",
        "opens_a_block",
        "fills_a_block",
        "first_block",
        "block_per_token",
        "second_block",
    ],
)
def test_hook_attends_over_the_pool_including_the_new_token(block_size, cached):
    """The written position must be visible to the very step that wrote it.

    Parametrised over where in a block that position lands, because the tail
    of the final block is allocated but unwritten: reading it as though it
    held real keys, or stopping one position short of the new token, are the
    two ways a block-table walk goes wrong at a boundary.
    """
    check_hook_against_dense(
        PagedKVCache(num_layers=1, num_blocks=16, block_size=block_size), cached
    )


def test_hook_follows_a_block_table_it_did_not_allocate_first():
    """Blocks are handed out from a shared pool, so ids are not positions."""
    check_hook_against_dense(
        PagedKVCache(num_layers=1, num_blocks=16, block_size=4), cached=5, decoys=2
    )


@pytest.mark.parametrize(
    ("batch", "q_len", "kv_heads"),
    [(2, 1, 2), (1, 3, 2), (1, 1, 1)],
    ids=["batched", "prefill", "grouped_query"],
)
def test_hook_defers_shapes_the_kernel_does_not_cover(
    monkeypatch, batch, q_len, kv_heads
):
    """Unsupported shapes go to the model's own attention, not to an error."""
    deferred = []
    monkeypatch.setattr(
        attention_hook,
        "sdpa_attention_forward",
        lambda *args, **kwargs: (deferred.append(args) or (None, None)),
    )

    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4)
    cache.add_sequence(0)
    cache.append(0, [kv_sequence(2, 4, 4)])
    position = cache.reserve(0, 1)

    query = torch.randn(batch, 2, q_len, 4)
    key = torch.randn(batch, kv_heads, q_len, 4)
    with paged_decode(cache, 0, position):
        paged_attention_forward(FakeAttention(), query, key, key, None, scaling=0.5)

    assert len(deferred) == 1


def test_hook_defers_when_no_sequence_is_being_decoded(monkeypatch):
    """Prefill and the batching path run through the same registered function."""
    deferred = []
    monkeypatch.setattr(
        attention_hook,
        "sdpa_attention_forward",
        lambda *args, **kwargs: (deferred.append(args) or (None, None)),
    )

    query = torch.randn(1, 2, 1, 4)
    paged_attention_forward(FakeAttention(), query, query, query, None, scaling=0.5)

    assert attention_hook.active_context() is None
    assert len(deferred) == 1


# -- the cache API the hook writes through ----------------------------------


def test_reserve_commits_blocks_and_hands_back_the_position():
    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4)
    cache.add_sequence(0)

    assert cache.reserve(0, 3) == 0
    assert cache.allocator.used_blocks == 1
    assert cache.reserve(0, 1) == 3
    assert cache.reserve(0, 1) == 4
    assert cache.allocator.used_blocks == 2
    assert cache.length(0) == 5


def test_written_positions_round_trip_through_gather():
    """`write` and `append` must lay tokens out identically.

    They are two ways into the same pool -- prefill uses one, decode the other
    -- so a sequence built by both has to read back as one sequence.
    """
    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4)
    cache.add_sequence(0)

    keys, values = kv_sequence(2, 5, 3)
    cache.append(0, [(keys[:, :, :3, :], values[:, :, :3, :])])
    for position in range(3, 5):
        assert cache.reserve(0, 1) == position
        cache.write(0, 0, position, keys[0, :, position, :], values[0, :, position, :])

    got_keys, got_values = cache.gather(0)[0]
    assert torch.equal(got_keys, keys)
    assert torch.equal(got_values, values)


def test_write_rejects_an_unreserved_position():
    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4)
    cache.add_sequence(0)
    cache.reserve(0, 1)

    with pytest.raises(ValueError):
        cache.write(0, 0, 1, torch.zeros(2, 3), torch.zeros(2, 3))


def test_reserve_rejects_a_non_positive_count():
    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4)
    cache.add_sequence(0)

    with pytest.raises(ValueError):
        cache.reserve(0, 0)


def test_block_table_tensor_indexes_the_pool():
    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4)
    cache.add_sequence(0)
    cache.append(0, [kv_sequence(2, 6, 3)])

    table = cache.block_table_tensor(0)

    assert table.dtype == torch.long
    assert table.tolist() == cache.block_table(0)
    assert cache.layer_keys(0)[table].shape[0] == len(cache.block_table(0))
