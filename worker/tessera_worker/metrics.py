"""Prometheus metrics: the telemetry the engines already compute, published.

The engines keep numbers that decide operational questions -- block-pool
occupancy, scheduler queue depth, speculative acceptance rate -- and until now
every one of them died with the process. This module exports them, and adds
the per-request totals no engine has a reason to keep for itself.

Two rules shape the whole file.

Instrumentation observes and nothing else. Nothing here may influence what a
decode produces, and every public entry point swallows its own exceptions: a
broken counter must never fail a request. Labels stay off the request-shaped
dimensions -- no prompt, no request id, no generated text -- because a metric
with unbounded cardinality is a slower outage than no metric at all.

State that a live object already owns is read at scrape time by a collector,
not mirrored into gauges from the request path. That keeps a scrape from
reporting occupancy that was true ten minutes ago, and it keeps metrics code
out of the decode loop this project holds byte-identical to the reference
engine.

Deliberately not here: anything per-model or per-request-id, and any metric
that would need the engines to record something they do not already track.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram
from prometheus_client import REGISTRY as _REGISTRY
from prometheus_client import generate_latest as _generate_latest
from prometheus_client import start_http_server as _start_http_server
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    InfoMetricFamily,
)

logger = logging.getLogger(__name__)

CONTENT_TYPE = CONTENT_TYPE_LATEST

# -- cumulative request totals ----------------------------------------------
# These are the only values the request path writes. Everything else is read
# off live objects when a scrape arrives.

REQUESTS = Counter(
    "tessera_requests",
    "Generation requests that reached a terminal state.",
    ["outcome"],
)
PROMPT_TOKENS = Counter(
    "tessera_prompt_tokens",
    "Prompt tokens accepted by the engine.",
)
GENERATED_TOKENS = Counter(
    "tessera_generated_tokens",
    "Tokens produced by the engine.",
)
REQUEST_LATENCY = Histogram(
    "tessera_request_latency_seconds",
    "End-to-end generation latency.",
    # Wider than the client default's 10s ceiling: a long completion on CPU
    # runs for minutes, and everything past the last bucket is invisible.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
)


def record_request(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_seconds: float | None = None,
    outcome: str = "success",
) -> None:
    """Record one finished request.

    `outcome` is a closed set -- "success", "error", "rejected" -- so the
    label cannot grow a new value per failure mode. `latency_seconds` is None
    for requests that never reached the model, which keeps rejections out of
    the latency histogram instead of dragging its low buckets down.
    """
    try:
        REQUESTS.labels(outcome=outcome).inc()
        if prompt_tokens:
            PROMPT_TOKENS.inc(prompt_tokens)
        if completion_tokens:
            GENERATED_TOKENS.inc(completion_tokens)
        if latency_seconds is not None:
            REQUEST_LATENCY.observe(latency_seconds)
    except Exception:  # noqa: BLE001 - metrics must never fail a request
        logger.exception("failed to record request metrics")


# -- live sources ------------------------------------------------------------


class _Sources:
    """The live objects a scrape reads through.

    Held weakly throughout. A process serves one engine, so the handle is
    last-bind-wins; servicers are a set because nothing forbids two, and
    in-flight counts add up across them where cache occupancy would not.
    """

    def __init__(self) -> None:
        self._handle: weakref.ref | None = None
        self._servicers: weakref.WeakSet = weakref.WeakSet()

    def bind_engine(self, handle: Any) -> None:
        self._handle = weakref.ref(handle)

    def track_servicer(self, servicer: Any) -> None:
        self._servicers.add(servicer)

    @property
    def handle(self) -> Any | None:
        return self._handle() if self._handle is not None else None

    @property
    def in_flight(self) -> int | None:
        """Requests currently executing, or None if nothing reports them."""
        servicers = list(self._servicers)
        if not servicers:
            return None
        return sum(s.in_flight for s in servicers)


_sources = _Sources()


def bind_engine(handle: Any) -> None:
    """Point the state gauges at the engine handle this process loaded."""
    try:
        _sources.bind_engine(handle)
    except Exception:  # noqa: BLE001 - metrics must never fail a load
        logger.exception("failed to bind engine to metrics")


def track_servicer(servicer: Any) -> None:
    """Register a gRPC servicer as a source for the in-flight gauge."""
    try:
        _sources.track_servicer(servicer)
    except Exception:  # noqa: BLE001 - metrics must never fail startup
        logger.exception("failed to track servicer for metrics")


def _paged_engine(engine: Any) -> Any | None:
    """The engine holding the block cache, if this backend has one.

    `BatchedEngine` wraps a `PagedEngine` rather than inheriting one, so the
    cache lives one level down; `reference` has no cache at all.
    """
    if engine is None:
        return None
    if getattr(engine, "cache", None) is not None:
        return engine
    inner = getattr(engine, "engine", None)
    return inner if getattr(inner, "cache", None) is not None else None


# -- scrape-time state -------------------------------------------------------


class _StateCollector:
    """Yields instantaneous engine state, computed when the scrape arrives.

    Metrics appear only when a source for them exists: a `reference` backend
    publishes no cache gauges rather than publishing zeros that look like an
    idle cache.
    """

    def collect(self) -> Iterator[Any]:
        try:
            yield from self._collect()
        except Exception:  # noqa: BLE001 - a bad scrape must not kill serving
            logger.exception("metrics collection failed")

    def _collect(self) -> Iterator[Any]:
        in_flight = _sources.in_flight
        if in_flight is not None:
            yield GaugeMetricFamily(
                "tessera_requests_in_flight",
                "Requests currently executing on the worker.",
                value=in_flight,
            )

        handle = _sources.handle
        if handle is None:
            return

        yield GaugeMetricFamily(
            "tessera_engine_ready",
            "1 when the engine has finished loading its model.",
            value=1.0 if handle.ready else 0.0,
        )
        yield InfoMetricFamily(
            "tessera_worker",
            "Static identity of this worker.",
            value={
                "backend": str(handle.backend),
                "model": str(handle.config.model_name),
                "device": str(handle.config.device),
            },
        )
        if not handle.ready:
            return

        engine = handle.engine
        yield from self._cache_metrics(_paged_engine(engine))
        yield from self._scheduler_metrics(getattr(engine, "batcher", None))
        yield from self._speculative_metrics(engine)

    def _cache_metrics(self, paged: Any) -> Iterator[Any]:
        if paged is None:
            return
        cache = paged.cache
        allocator = cache.allocator
        yield GaugeMetricFamily(
            "tessera_kv_cache_blocks_total",
            "Blocks in the paged KV pool.",
            value=allocator.num_blocks,
        )
        yield GaugeMetricFamily(
            "tessera_kv_cache_blocks_used",
            "Blocks currently held by sequences.",
            value=allocator.used_blocks,
        )
        yield GaugeMetricFamily(
            "tessera_kv_cache_blocks_free",
            "Blocks available for admission.",
            value=allocator.free_blocks,
        )
        yield GaugeMetricFamily(
            "tessera_kv_cache_utilisation",
            "Fraction of the block pool in use, 0 to 1.",
            value=cache.utilisation,
        )

    def _scheduler_metrics(self, batcher: Any) -> Iterator[Any]:
        if batcher is None:
            return
        yield GaugeMetricFamily(
            "tessera_scheduler_waiting_requests",
            "Requests admitted to the queue but not yet decoding.",
            value=len(batcher.waiting),
        )
        yield GaugeMetricFamily(
            "tessera_scheduler_running_sequences",
            "Sequences the scheduler advances each step.",
            value=len(batcher.running),
        )
        yield GaugeMetricFamily(
            "tessera_scheduler_pending_requests",
            "Waiting plus running: everything the scheduler still owes.",
            value=batcher.pending,
        )
        yield GaugeMetricFamily(
            "tessera_scheduler_max_batch_size",
            "Slot limit on the batch width.",
            value=batcher.max_batch_size,
        )

    def _speculative_metrics(self, engine: Any) -> Iterator[Any]:
        if not hasattr(engine, "acceptance_rate"):
            return
        # Read off the engine rather than counted on the request path: the
        # engine already increments these at verification time, and a second
        # counter would be a second thing to keep in step.
        yield CounterMetricFamily(
            "tessera_speculative_proposed_tokens",
            "Draft tokens the target model has judged.",
            value=engine.proposed_tokens,
        )
        yield CounterMetricFamily(
            "tessera_speculative_accepted_tokens",
            "Draft tokens accepted by the target model.",
            value=engine.accepted_tokens,
        )
        yield GaugeMetricFamily(
            "tessera_speculative_acceptance_rate",
            "Accepted over proposed, 0 when nothing has been proposed.",
            value=engine.acceptance_rate,
        )


_REGISTRY.register(_StateCollector())


# -- exposition --------------------------------------------------------------


def render() -> bytes:
    """The registry in Prometheus text exposition format."""
    return _generate_latest(_REGISTRY)


def start_metrics_server(port: int) -> Any | None:
    """Serve the registry over HTTP on `port`. Returns the server, or None.

    Opt-in: a port of 0 or less does nothing. The gRPC worker is also run
    embedded -- by the test suite and the benchmark harness -- and neither
    should discover it has opened a listening socket. A port already in use
    is logged and skipped rather than raised, because failing to export
    metrics is not a reason to fail to serve.

    The server is returned so a caller that owns its lifetime (a test, mostly)
    can shut it down; `serve` ignores it and lets the daemon thread die with
    the process.
    """
    if port <= 0:
        return None
    try:
        server, _thread = _start_http_server(port)
    except OSError:
        logger.warning("metrics server could not bind port %d", port, exc_info=True)
        return None
    logger.info("metrics listening on http://0.0.0.0:%d/metrics", port)
    return server
