"""Phase 2: the continuous batcher wired into the serving path.

The scheduler already had unit tests driving `step` directly. These cover what
production actually does: several threads issuing requests at once against one
engine, merged into shared forward passes. Concurrency must not change a single
token, so every result is still checked against the reference engine.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tessera_worker.engine import BACKENDS, EngineHandle
from tessera_worker.sampling import SamplingParams
from tessera_worker.serving import BatchedEngine

PROMPTS = [
    "The capital of France is",
    "Hello world",
    "Once upon a time",
    "The quick brown fox jumps over",
]


@pytest.fixture(scope="module")
def batched(config):
    engine = BatchedEngine(config, max_batch_size=4, num_blocks=256, block_size=8)
    yield engine
    engine.close()


# -- registration -----------------------------------------------------------


def test_batched_backend_is_registered():
    assert BACKENDS["batched"] is BatchedEngine


def test_handle_can_load_the_batched_backend(config):
    handle = EngineHandle(config, backend="batched")
    handle.load()
    try:
        assert handle.ready
        assert (
            handle.engine.generate("Hello world", max_tokens=2).completion_tokens == 2
        )
    finally:
        handle.engine.close()


# -- single request ---------------------------------------------------------


@pytest.mark.parametrize("prompt", PROMPTS, ids=["capital", "hello", "once", "fox"])
def test_single_request_matches_reference(batched, engine, prompt):
    expected = engine.generate(prompt, max_tokens=6)
    actual = batched.generate(prompt, max_tokens=6)

    assert actual.token_ids == expected.token_ids
    assert actual.text == expected.text


def test_reports_token_accounting(batched, engine):
    expected = engine.generate("Hello world", max_tokens=5)
    actual = batched.generate("Hello world", max_tokens=5)

    assert actual.prompt_tokens == expected.prompt_tokens
    assert actual.completion_tokens == expected.completion_tokens
    assert actual.latency_ms >= 0.0


def test_streaming_deltas_reassemble(batched):
    chunks = list(batched.iter_generate("Hello world", max_tokens=6))
    streamed = "".join(c.text for c in chunks)

    assert streamed == batched.generate("Hello world", max_tokens=6).text
    assert chunks[-1].finished is True


# -- concurrency ------------------------------------------------------------


def test_concurrent_requests_match_reference(batched, engine):
    """The point of the whole exercise: sharing a pass changes no output."""
    expected = {p: engine.generate(p, max_tokens=6).token_ids for p in PROMPTS}

    with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        futures = {
            prompt: pool.submit(batched.generate, prompt, 6) for prompt in PROMPTS
        }
        actual = {
            prompt: f.result(timeout=120).token_ids for prompt, f in futures.items()
        }

    assert actual == expected


def test_more_requests_than_batch_slots_all_complete(batched, engine):
    """Oversubscription must queue, not drop or deadlock."""
    prompts = PROMPTS * 3
    expected = engine.generate("Hello world", max_tokens=4).token_ids

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(batched.generate, p, 4) for p in prompts]
        results = [f.result(timeout=180) for f in futures]

    assert len(results) == len(prompts)
    assert all(r.completion_tokens == 4 for r in results)
    # The repeated prompt must decode identically every time it appears.
    for result, prompt in zip(results, prompts, strict=True):
        if prompt == "Hello world":
            assert result.token_ids == expected


def test_concurrent_streaming_does_not_interleave_requests(batched):
    """Each stream must carry only its own tokens."""

    def collect(prompt: str) -> str:
        return "".join(c.text for c in batched.iter_generate(prompt, max_tokens=5))

    with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        streamed = list(pool.map(collect, PROMPTS))

    for prompt, text in zip(PROMPTS, streamed, strict=True):
        assert text == batched.generate(prompt, max_tokens=5).text


def test_blocks_are_returned_after_concurrent_load(batched):
    before = batched.engine.cache.allocator.free_blocks

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda p: batched.generate(p, 4), PROMPTS))

    assert batched.engine.cache.allocator.free_blocks == before


def test_seeded_sampling_matches_reference(batched, engine):
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=11)

    expected = engine.generate("Once upon a time", max_tokens=5, params=params)
    actual = batched.generate("Once upon a time", max_tokens=5, params=params)

    assert actual.token_ids == expected.token_ids


# -- shutdown ---------------------------------------------------------------


def test_close_is_idempotent(config):
    engine = BatchedEngine(config, max_batch_size=2, num_blocks=32, block_size=8)
    engine.close()
    engine.close()

    assert not engine._thread.is_alive()


def test_requests_are_refused_after_close(config):
    engine = BatchedEngine(config, max_batch_size=2, num_blocks=32, block_size=8)
    engine.close()

    with pytest.raises(RuntimeError):
        engine.generate("Hello world", max_tokens=2)


def test_context_manager_stops_the_scheduler(config):
    with BatchedEngine(config, max_batch_size=2, num_blocks=64, block_size=8) as engine:
        assert engine.generate("Hello world", max_tokens=2).completion_tokens == 2

    assert not engine._thread.is_alive()


def test_a_failing_step_fails_waiting_requests(config, monkeypatch):
    """A scheduler crash must surface, not hang every caller on a queue."""
    engine = BatchedEngine(config, max_batch_size=2, num_blocks=64, block_size=8)

    def boom():
        raise RuntimeError("scheduler exploded")

    monkeypatch.setattr(engine.batcher, "step", boom)
    try:
        with pytest.raises(RuntimeError, match="scheduler exploded"):
            engine.generate("Hello world", max_tokens=4)
    finally:
        engine.close()
