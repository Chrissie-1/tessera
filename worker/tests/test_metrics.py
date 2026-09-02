"""Metrics, checked against the state they claim to describe.

A test that asserts `/metrics` returns 200 proves nothing: it passes just as
well when every counter is frozen at zero and the collector is wired to
nothing. So each test here reads a number out of the exposition text and
compares it against the thing that produced it -- the response the servicer
returned, the allocator's own block count, the scheduler's own queue -- and
fails if the two ever stop agreeing.

The engines are not exercised for correctness here; that is what the rest of
the suite is for. What is exercised is the wiring, plus the promise that a
broken metric degrades into a missing number rather than a failed request.
"""

from __future__ import annotations

import socket
import threading
import urllib.request
from dataclasses import dataclass

import pytest
import torch
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from tessera_worker import metrics
from tessera_worker.api import create_app
from tessera_worker.batching import Request as BatchRequest
from tessera_worker.config import WorkerConfig
from tessera_worker.engine import EngineHandle
from tessera_worker.generated import inference_pb2
from tessera_worker.model import GenerationResult
from tessera_worker.server import InferenceServicer

PROMPT = "The capital of France is"


# -- reading the exposition text ---------------------------------------------


def scrape() -> dict[tuple[str, tuple], float]:
    """Every sample in the registry, keyed by name and labels."""
    text = metrics.render().decode("utf-8")
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    }


def value(samples: dict, name: str, **labels: str) -> float:
    """One sample, defaulting to 0 for a series that has not appeared yet."""
    return samples.get((name, tuple(sorted(labels.items()))), 0.0)


def names(samples: dict) -> set[str]:
    return {name for name, _ in samples}


# -- doubles ------------------------------------------------------------------


class AbortedError(Exception):
    def __init__(self, code, details: str) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


class FakeContext:
    def abort(self, code, details):
        raise AbortedError(code, details)


@dataclass
class FakeHandle:
    """The subset of EngineHandle the servicer touches."""

    config: WorkerConfig
    engine: object
    ready: bool = True
    backend: str = "paged"


class ExplodingEngine:
    def generate(self, **_):
        raise RuntimeError("CUDA out of memory")


class BlockingEngine:
    """Parks inside `generate` so a scrape can observe a request in flight."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, **_):
        self.entered.set()
        self.release.wait(timeout=10)
        return GenerationResult(text="", prompt_tokens=1, completion_tokens=0)


def grpc_request(**overrides):
    payload = {"prompt": PROMPT, "max_tokens": 4}
    payload.update(overrides)
    return inference_pb2.GenerateRequest(**payload)


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def paged_handle(config) -> EngineHandle:
    handle = EngineHandle(config, backend="paged")
    handle.load()
    return handle


@pytest.fixture(scope="module")
def speculative_handle(config) -> EngineHandle:
    handle = EngineHandle(config, backend="speculative")
    handle.load()
    return handle


@pytest.fixture(scope="module")
def batched_handle(config) -> EngineHandle:
    handle = EngineHandle(config, backend="batched")
    handle.load()
    yield handle
    handle.engine.close()


@pytest.fixture
def bind():
    """Select which loaded handle the state gauges read from.

    `EngineHandle.load` does this by itself in production; tests bind
    explicitly because several handles are alive at once and the collector
    describes exactly one.
    """

    def _bind(handle: EngineHandle) -> EngineHandle:
        metrics.bind_engine(handle)
        return handle

    return _bind


# -- request totals -----------------------------------------------------------


def test_generate_moves_the_token_counters_by_what_it_returned(config, engine):
    servicer = InferenceServicer(FakeHandle(config=config, engine=engine))
    before = scrape()

    response = servicer.Generate(grpc_request(max_tokens=4), FakeContext())

    after = scrape()
    assert response.completion_tokens > 0
    assert value(after, "tessera_generated_tokens_total") - value(
        before, "tessera_generated_tokens_total"
    ) == pytest.approx(response.completion_tokens)
    assert value(after, "tessera_prompt_tokens_total") - value(
        before, "tessera_prompt_tokens_total"
    ) == pytest.approx(response.prompt_tokens)
    assert value(after, "tessera_requests_total", outcome="success") - value(
        before, "tessera_requests_total", outcome="success"
    ) == pytest.approx(1.0)


def test_latency_histogram_observes_one_sample_per_request(config, engine):
    servicer = InferenceServicer(FakeHandle(config=config, engine=engine))
    before = scrape()

    servicer.Generate(grpc_request(max_tokens=2), FakeContext())

    after = scrape()
    assert value(after, "tessera_request_latency_seconds_count") - value(
        before, "tessera_request_latency_seconds_count"
    ) == pytest.approx(1.0)
    # A decode takes real time, so the sum has to have moved with it.
    assert value(after, "tessera_request_latency_seconds_sum") > value(
        before, "tessera_request_latency_seconds_sum"
    )


def test_a_rejected_request_is_counted_but_not_timed(config, engine):
    servicer = InferenceServicer(FakeHandle(config=config, engine=engine))
    before = scrape()

    with pytest.raises(AbortedError):
        servicer.Generate(
            grpc_request(max_tokens=config.max_tokens_cap + 1), FakeContext()
        )

    after = scrape()
    assert value(after, "tessera_requests_total", outcome="rejected") - value(
        before, "tessera_requests_total", outcome="rejected"
    ) == pytest.approx(1.0)
    # It never reached the model, so it must not appear in the histogram.
    assert value(after, "tessera_request_latency_seconds_count") == pytest.approx(
        value(before, "tessera_request_latency_seconds_count")
    )
    assert value(after, "tessera_generated_tokens_total") == pytest.approx(
        value(before, "tessera_generated_tokens_total")
    )


def test_a_failed_generation_is_counted_as_an_error(config):
    servicer = InferenceServicer(FakeHandle(config=config, engine=ExplodingEngine()))
    before = scrape()

    with pytest.raises(AbortedError):
        servicer.Generate(grpc_request(), FakeContext())

    after = scrape()
    assert value(after, "tessera_requests_total", outcome="error") - value(
        before, "tessera_requests_total", outcome="error"
    ) == pytest.approx(1.0)
    assert value(after, "tessera_requests_total", outcome="success") == pytest.approx(
        value(before, "tessera_requests_total", outcome="success")
    )


def test_a_stream_counts_every_token_it_yielded(config, paged_handle):
    servicer = InferenceServicer(
        FakeHandle(config=config, engine=paged_handle.engine, backend="paged")
    )
    before = scrape()

    responses = list(servicer.GenerateStream(grpc_request(max_tokens=5), FakeContext()))

    after = scrape()
    assert responses
    streamed = responses[-1].completion_tokens
    assert streamed > 0
    assert value(after, "tessera_generated_tokens_total") - value(
        before, "tessera_generated_tokens_total"
    ) == pytest.approx(streamed)


# -- instantaneous state ------------------------------------------------------


def test_in_flight_gauge_follows_a_request_in_progress(config):
    blocking = BlockingEngine()
    servicer = InferenceServicer(FakeHandle(config=config, engine=blocking))
    baseline = value(scrape(), "tessera_requests_in_flight")

    worker = threading.Thread(
        target=lambda: servicer.Generate(grpc_request(), FakeContext())
    )
    worker.start()
    try:
        assert blocking.entered.wait(timeout=10)
        during = value(scrape(), "tessera_requests_in_flight")
    finally:
        blocking.release.set()
        worker.join(timeout=10)

    assert during == pytest.approx(baseline + 1)
    assert servicer.in_flight == 0
    assert value(scrape(), "tessera_requests_in_flight") == pytest.approx(baseline)


def test_cache_gauges_track_live_block_occupancy(bind, paged_handle):
    engine = bind(paged_handle).engine
    stream = engine.iter_generate(PROMPT, max_tokens=8)
    next(stream)
    next(stream)
    try:
        samples = scrape()
        used = value(samples, "tessera_kv_cache_blocks_used")
        free = value(samples, "tessera_kv_cache_blocks_free")
        total = value(samples, "tessera_kv_cache_blocks_total")
        utilisation = value(samples, "tessera_kv_cache_utilisation")

        # Mid-decode the sequence holds blocks; the gauges have to say so.
        assert used == pytest.approx(engine.cache.allocator.used_blocks)
        assert used >= 1
        assert total == pytest.approx(engine.cache.allocator.num_blocks)
        assert used + free == pytest.approx(total)
        assert utilisation == pytest.approx(engine.cache.utilisation)
        assert 0.0 < utilisation <= 1.0
    finally:
        stream.close()

    freed = scrape()
    assert value(freed, "tessera_kv_cache_blocks_used") == pytest.approx(0.0)
    assert value(freed, "tessera_kv_cache_utilisation") == pytest.approx(0.0)


def test_cache_gauges_report_the_configured_pool_size(config, bind):
    sized = WorkerConfig(
        model_name=config.model_name,
        device="cpu",
        dtype=torch.float32,
        grpc_port=0,
        http_port=0,
        max_tokens_cap=16,
        num_blocks=37,
        block_size=8,
    )
    handle = EngineHandle(sized, backend="paged")
    handle.load()
    bind(handle)

    assert value(scrape(), "tessera_kv_cache_blocks_total") == pytest.approx(37.0)


def test_scheduler_gauges_report_the_real_queue(bind, batched_handle):
    engine = bind(batched_handle).engine
    # Stop the scheduler thread first, so the queue this asserts on cannot
    # drain underneath the scrape. The batcher being read is the real one.
    engine.close()
    before = scrape()

    for index in range(3):
        engine.batcher.submit(
            BatchRequest(request_id=f"metrics-{index}", prompt=PROMPT, max_tokens=2)
        )

    after = scrape()
    assert value(after, "tessera_scheduler_waiting_requests") - value(
        before, "tessera_scheduler_waiting_requests"
    ) == pytest.approx(3.0)
    assert value(after, "tessera_scheduler_pending_requests") == pytest.approx(
        engine.batcher.pending
    )
    assert value(after, "tessera_scheduler_running_sequences") == pytest.approx(
        len(engine.batcher.running)
    )
    assert value(after, "tessera_scheduler_max_batch_size") == pytest.approx(
        engine.batcher.max_batch_size
    )


def test_speculative_counters_come_from_the_engine(bind, speculative_handle):
    engine = bind(speculative_handle).engine
    engine.generate(PROMPT, max_tokens=8)

    samples = scrape()
    assert engine.proposed_tokens > 0
    assert value(samples, "tessera_speculative_proposed_tokens_total") == pytest.approx(
        engine.proposed_tokens
    )
    assert value(samples, "tessera_speculative_accepted_tokens_total") == pytest.approx(
        engine.accepted_tokens
    )
    assert value(samples, "tessera_speculative_acceptance_rate") == pytest.approx(
        engine.acceptance_rate
    )


def test_a_backend_without_a_cache_publishes_no_cache_gauges(config, bind):
    handle = EngineHandle(config, backend="reference")
    handle.load()
    bind(handle)

    samples = scrape()
    assert "tessera_kv_cache_blocks_total" not in names(samples)
    assert "tessera_scheduler_waiting_requests" not in names(samples)
    assert "tessera_speculative_acceptance_rate" not in names(samples)
    assert value(samples, "tessera_engine_ready") == pytest.approx(1.0)


def test_loading_an_engine_binds_it_without_being_asked(config):
    """Production never calls `bind_engine`; `load` does it."""
    handle = EngineHandle(config, backend="reference")
    handle.load()

    assert (
        value(
            scrape(),
            "tessera_worker_info",
            backend="reference",
            model=config.model_name,
            device=config.device,
        )
        == 1.0
    )


# -- exposition surfaces ------------------------------------------------------


def test_http_metrics_endpoint_reflects_completions_it_served(config):
    with TestClient(create_app(config)) as client:
        before = scrape()
        body = client.post(
            "/v1/completions", json={"prompt": PROMPT, "max_tokens": 3}
        ).json()

        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

        after = scrape()
        assert body["completion_tokens"] > 0
        assert value(after, "tessera_generated_tokens_total") - value(
            before, "tessera_generated_tokens_total"
        ) == pytest.approx(body["completion_tokens"])
        # The endpoint serves the same registry the test parses.
        assert "tessera_generated_tokens_total" in response.text


def test_metrics_server_stays_off_without_a_port(monkeypatch):
    started: list[int] = []
    monkeypatch.setattr(
        metrics, "_start_http_server", lambda port: started.append(port)
    )

    assert metrics.start_metrics_server(0) is None
    assert metrics.start_metrics_server(-1) is None
    assert started == []


def test_metrics_server_survives_a_port_it_cannot_bind(monkeypatch):
    def refuse(port):
        raise OSError("address already in use")

    monkeypatch.setattr(metrics, "_start_http_server", refuse)

    assert metrics.start_metrics_server(9999) is None


def test_metrics_server_serves_the_registry_over_http():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = metrics.start_metrics_server(port)
    assert server is not None
    try:
        url = f"http://127.0.0.1:{port}/metrics"
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()

    assert "tessera_request_latency_seconds_bucket" in body


# -- failure containment ------------------------------------------------------


def test_a_broken_counter_does_not_fail_a_request(config, engine, monkeypatch):
    class Sabotaged:
        def labels(self, **_):
            raise RuntimeError("metrics backend exploded")

    monkeypatch.setattr(metrics, "REQUESTS", Sabotaged())
    servicer = InferenceServicer(FakeHandle(config=config, engine=engine))

    response = servicer.Generate(grpc_request(max_tokens=2), FakeContext())

    assert response.finished is True
    assert response.completion_tokens > 0


def test_a_broken_collector_does_not_break_the_scrape(bind, paged_handle):
    class Hostile:
        @property
        def ready(self):
            raise RuntimeError("engine state is unreadable")

    hostile = Hostile()
    bind(hostile)
    try:
        body = metrics.render().decode("utf-8")
    finally:
        bind(paged_handle)

    # The collector fails closed: its own metrics vanish, the rest survive.
    assert "tessera_request_latency_seconds_bucket" in body
    assert "tessera_kv_cache_blocks_total" not in body
