"""Phase 2: block allocator, paged cache, and paged decoding.

The equivalence tests at the bottom are the ones that matter. A cache bug
shows up as a divergent token, so asserting the paged engine reproduces the
reference engine exactly is what licenses using it in production.
"""

from __future__ import annotations

import pytest
import torch

from tessera_worker.paged import (
    BlockAllocator,
    OutOfBlocksError,
    PagedKVCache,
)
from tessera_worker.sampling import SamplingParams


@pytest.fixture(scope="module")
def paged(config):
    from tessera_worker.paged_engine import PagedEngine

    # A deliberately small pool with a small block size, so the tests cross
    # block boundaries instead of living inside block zero.
    return PagedEngine(config, num_blocks=64, block_size=4)


def kv(num_layers: int, tokens: int, heads: int = 2, dim: int = 3):
    """Per-layer (keys, values) with distinct values per position."""
    out = []
    for layer in range(num_layers):
        base = torch.arange(tokens, dtype=torch.float32).reshape(1, 1, tokens, 1)
        keys = base.expand(1, heads, tokens, dim) + layer * 100.0
        out.append((keys.clone(), keys.clone() + 0.5))
    return out


# -- allocator --------------------------------------------------------------


def test_allocator_hands_out_distinct_blocks():
    allocator = BlockAllocator(4)
    blocks = [allocator.allocate() for _ in range(4)]

    assert sorted(blocks) == [0, 1, 2, 3]
    assert allocator.free_blocks == 0


def test_allocator_raises_when_exhausted():
    allocator = BlockAllocator(1)
    allocator.allocate()

    with pytest.raises(OutOfBlocksError):
        allocator.allocate()


def test_freed_blocks_are_reusable():
    allocator = BlockAllocator(2)
    first = allocator.allocate()
    allocator.free([first])

    assert allocator.free_blocks == 2
    assert allocator.allocate() == first


def test_double_free_is_rejected():
    """A block returned twice would later be handed to two sequences."""
    allocator = BlockAllocator(2)
    block = allocator.allocate()
    allocator.free([block])

    with pytest.raises(ValueError):
        allocator.free([block])


def test_allocator_rejects_empty_pool():
    with pytest.raises(ValueError):
        BlockAllocator(0)


# -- cache ------------------------------------------------------------------


def test_blocks_are_committed_only_as_needed():
    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4)
    cache.add_sequence(0)
    assert cache.allocator.used_blocks == 0

    cache.append(0, kv(1, 3))
    assert cache.allocator.used_blocks == 1

    # Crossing into a second block commits exactly one more.
    cache.append(0, kv(1, 2))
    assert cache.allocator.used_blocks == 2
    assert cache.length(0) == 5


def test_freeing_a_sequence_returns_every_block():
    cache = PagedKVCache(num_layers=1, num_blocks=8, block_size=4)
    cache.add_sequence(0)
    cache.append(0, kv(1, 9))
    assert cache.allocator.used_blocks == 3

    cache.free_sequence(0)
    assert cache.allocator.free_blocks == 8
    assert not cache.has_sequence(0)


def test_free_is_idempotent():
    cache = PagedKVCache(num_layers=1, num_blocks=4, block_size=4)
    cache.add_sequence(0)
    cache.append(0, kv(1, 2))

    cache.free_sequence(0)
    cache.free_sequence(0)

    assert cache.allocator.free_blocks == 4


def test_gather_round_trips_across_block_boundaries():
    """What comes out must equal what went in, regardless of block layout."""
    cache = PagedKVCache(num_layers=2, num_blocks=16, block_size=4)
    cache.add_sequence(0)

    written = kv(2, 10)
    cache.append(0, written)
    gathered = cache.gather(0)

    for (want_k, want_v), (got_k, got_v) in zip(written, gathered, strict=True):
        assert torch.equal(got_k, want_k)
        assert torch.equal(got_v, want_v)


def test_gather_is_contiguous_across_appends():
    """Decoding appends one token per step; that must match a single write."""
    bulk = PagedKVCache(num_layers=1, num_blocks=16, block_size=4)
    bulk.add_sequence(0)
    written = kv(1, 6)
    bulk.append(0, written)

    stepwise = PagedKVCache(num_layers=1, num_blocks=16, block_size=4)
    stepwise.add_sequence(0)
    keys, values = written[0]
    for position in range(6):
        stepwise.append(
            0,
            [
                (
                    keys[:, :, position : position + 1, :],
                    values[:, :, position : position + 1, :],
                )
            ],
        )

    assert torch.equal(stepwise.gather(0)[0][0], bulk.gather(0)[0][0])
    assert torch.equal(stepwise.gather(0)[0][1], bulk.gather(0)[0][1])


def test_sequences_do_not_share_blocks():
    cache = PagedKVCache(num_layers=1, num_blocks=16, block_size=4)
    cache.add_sequence(0)
    cache.add_sequence(1)

    cache.append(0, kv(1, 5))
    cache.append(1, kv(1, 5))

    assert set(cache.block_table(0)).isdisjoint(cache.block_table(1))


def test_blocks_needed_predicts_allocation():
    cache = PagedKVCache(num_layers=1, num_blocks=16, block_size=4)
    cache.add_sequence(0)
    cache.append(0, kv(1, 3))

    # One slot left in the current block, so 5 more tokens need 1 more block.
    predicted = cache.blocks_needed(0, 5)
    before = cache.allocator.used_blocks
    cache.append(0, kv(1, 5))

    assert cache.allocator.used_blocks - before == predicted


def test_cache_reports_exhaustion_rather_than_evicting():
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=4)
    cache.add_sequence(0)

    with pytest.raises(OutOfBlocksError):
        cache.append(0, kv(1, 12))


def test_duplicate_sequence_is_rejected():
    cache = PagedKVCache(num_layers=1, num_blocks=4, block_size=4)
    cache.add_sequence(0)

    with pytest.raises(ValueError):
        cache.add_sequence(0)


def test_rejects_non_positive_block_size():
    with pytest.raises(ValueError):
        PagedKVCache(num_layers=1, num_blocks=4, block_size=0)


# -- equivalence with the reference engine ----------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "The capital of France is",
        "",
        "Numbers: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
    ],
    ids=["short", "empty", "spans_blocks"],
)
def test_paged_greedy_matches_reference(engine, paged, prompt):
    reference = engine.generate(prompt, max_tokens=12)
    actual = paged.generate(prompt, max_tokens=12)

    assert actual.token_ids == reference.token_ids
    assert actual.text == reference.text
    assert actual.finish_reason == reference.finish_reason


def test_paged_seeded_sampling_matches_reference(engine, paged):
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=42)

    reference = engine.generate("Once upon a time", max_tokens=8, params=params)
    actual = paged.generate("Once upon a time", max_tokens=8, params=params)

    assert actual.token_ids == reference.token_ids


def test_paged_token_accounting_matches_reference(engine, paged):
    reference = engine.generate("Hello world", max_tokens=6)
    actual = paged.generate("Hello world", max_tokens=6)

    assert actual.prompt_tokens == reference.prompt_tokens
    assert actual.completion_tokens == reference.completion_tokens


def test_blocks_are_returned_after_each_request(paged):
    before = paged.cache.allocator.free_blocks
    paged.generate("The capital of France is", max_tokens=8)

    assert paged.cache.allocator.free_blocks == before


def test_blocks_are_returned_even_when_generation_fails(paged, monkeypatch):
    before = paged.cache.allocator.free_blocks

    def boom(*_args, **_kwargs):
        raise RuntimeError("forward exploded")

    monkeypatch.setattr(paged, "model", boom)
    with pytest.raises(RuntimeError):
        paged.generate("anything", max_tokens=4)

    assert paged.cache.allocator.free_blocks == before


def test_repeated_requests_do_not_leak(paged):
    before = paged.cache.allocator.free_blocks
    for _ in range(3):
        paged.generate("Hello world", max_tokens=5)

    assert paged.cache.allocator.free_blocks == before
