"""Phase 2: continuous batching.

Batching changes when work runs, never what it produces. The equivalence tests
hold the scheduler to the reference engine's output: if left-padding, masking
or position ids were wrong, a batched sequence would drift from the answer the
same prompt gets on its own.
"""

from __future__ import annotations

import pytest

from tessera_worker.batching import ContinuousBatcher, Request
from tessera_worker.paged import PagedKVCache
from tessera_worker.sampling import SamplingParams


@pytest.fixture(scope="module")
def paged(config):
    from tessera_worker.paged_engine import PagedEngine

    return PagedEngine(config, num_blocks=256, block_size=8)


@pytest.fixture
def batcher(paged):
    # Reset the pool between tests so leak assertions start from a clean slate.
    paged.cache = PagedKVCache(
        num_layers=int(paged.model.config.n_layer),
        num_blocks=256,
        block_size=8,
        device=paged.config.device,
        dtype=paged.config.dtype,
    )
    return ContinuousBatcher(paged, max_batch_size=4)


def drain(batcher: ContinuousBatcher) -> dict[str, list[int]]:
    return batcher.run_to_completion()


# -- queue mechanics --------------------------------------------------------


def test_rejects_non_positive_batch_size(paged):
    with pytest.raises(ValueError):
        ContinuousBatcher(paged, max_batch_size=0)


def test_no_work_means_no_steps(batcher):
    assert batcher.has_work is False
    assert batcher.step() == []


def test_submitted_request_is_pending(batcher):
    batcher.submit(Request("a", "Hello world", max_tokens=2))

    assert batcher.pending == 1
    assert batcher.has_work is True


def test_admission_is_capped_by_batch_size(batcher):
    for i in range(6):
        batcher.submit(Request(str(i), "Hello world", max_tokens=8))
    batcher.step()

    assert len(batcher.running) == 4
    assert len(batcher.waiting) == 2


def test_each_sequence_advances_one_token_per_step(batcher):
    batcher.submit(Request("a", "Hello world", max_tokens=8))
    batcher.submit(Request("b", "The capital of France is", max_tokens=8))

    batcher.step()
    assert [len(s.tokens) for s in batcher.running] == [1, 1]

    batcher.step()
    assert [len(s.tokens) for s in batcher.running] == [2, 2]


# -- lifecycle --------------------------------------------------------------


def test_finished_sequences_are_retired(batcher):
    batcher.submit(Request("a", "Hello world", max_tokens=1))
    batcher.step()

    assert batcher.running == []
    assert batcher.has_work is False


def test_blocks_are_returned_when_a_sequence_finishes(batcher):
    before = batcher.engine.cache.allocator.free_blocks
    batcher.submit(Request("a", "Hello world", max_tokens=3))
    drain(batcher)

    assert batcher.engine.cache.allocator.free_blocks == before


def test_queued_work_is_admitted_as_slots_free(batcher):
    """The point of continuous batching: no waiting for the batch to drain."""
    for i in range(4):
        batcher.submit(Request(f"short{i}", "Hello world", max_tokens=1))
    batcher.submit(Request("late", "Hello world", max_tokens=2))

    batcher.step()
    # The four short requests finished in step one, freeing every slot.
    assert len(batcher.running) == 0
    assert len(batcher.waiting) == 1

    batcher.step()
    assert any(s.request_id == "late" for s in batcher.running)


def test_all_requests_complete_when_oversubscribed(batcher):
    for i in range(9):
        batcher.submit(Request(str(i), "Hello world", max_tokens=3))
    collected = drain(batcher)

    assert len(collected) == 9
    assert all(len(tokens) == 3 for tokens in collected.values())
    assert batcher.pending == 0


def test_draining_leaks_no_blocks(batcher):
    before = batcher.engine.cache.allocator.free_blocks
    for i in range(6):
        batcher.submit(Request(str(i), "The capital of France is", max_tokens=4))
    drain(batcher)

    assert batcher.engine.cache.allocator.free_blocks == before


def test_max_tokens_is_respected_exactly(batcher):
    batcher.submit(Request("a", "Hello world", max_tokens=5))
    collected = drain(batcher)

    assert len(collected["a"]) == 5


def test_step_reports_completion(batcher):
    batcher.submit(Request("a", "Hello world", max_tokens=1))
    outputs = batcher.step()

    assert len(outputs) == 1
    assert outputs[0].request_id == "a"
    assert outputs[0].finished is True
    assert outputs[0].finish_reason == "length"
    assert isinstance(outputs[0].text, str)


# -- backpressure -----------------------------------------------------------


def test_admission_defers_when_the_cache_is_full(paged):
    """A request that cannot get blocks waits; it must not evict a running one."""
    paged.cache = PagedKVCache(
        num_layers=int(paged.model.config.n_layer),
        num_blocks=2,
        block_size=8,
        device=paged.config.device,
        dtype=paged.config.dtype,
    )
    batcher = ContinuousBatcher(paged, max_batch_size=4)

    long_prompt = "word " * 40
    for i in range(3):
        batcher.submit(Request(str(i), long_prompt, max_tokens=2))
    batcher.step()

    assert len(batcher.running) <= 1
    assert batcher.waiting


# -- equivalence with single-sequence decoding ------------------------------


def test_batched_output_matches_unbatched(batcher, paged, engine):
    """Sequences decoded side by side must match decoding them alone."""
    prompts = {
        "a": "The capital of France is",
        "b": "Hello world",
        "c": "Once upon a time",
    }
    for request_id, prompt in prompts.items():
        batcher.submit(Request(request_id, prompt, max_tokens=6))
    collected = drain(batcher)

    for request_id, prompt in prompts.items():
        assert collected[request_id] == engine.generate(prompt, max_tokens=6).token_ids


def test_ragged_lengths_do_not_corrupt_neighbours(batcher, engine):
    """Left-padding is only correct if a short row cannot disturb a long one."""
    short = "Hi"
    long = "Numbers: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    batcher.submit(Request("short", short, max_tokens=6))
    batcher.submit(Request("long", long, max_tokens=6))
    collected = drain(batcher)

    assert collected["short"] == engine.generate(short, max_tokens=6).token_ids
    assert collected["long"] == engine.generate(long, max_tokens=6).token_ids


def test_seeded_sampling_matches_unbatched(batcher, engine):
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=7)
    batcher.submit(Request("a", "Once upon a time", max_tokens=5, params=params))
    collected = drain(batcher)

    expected = engine.generate("Once upon a time", max_tokens=5, params=params)
    assert collected["a"] == expected.token_ids
