"""gRPC server exposing the inference engine to the Rust gateway."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from concurrent import futures

import grpc

from . import metrics
from .config import WorkerConfig
from .engine import EngineHandle
from .generated import inference_pb2, inference_pb2_grpc
from .sampling import SamplingParams

logger = logging.getLogger(__name__)


def _params_from(request) -> SamplingParams:
    """proto3 leaves unset numerics at zero; top_p 0 means "unset", not "drop
    every token", so it is read as no filtering."""
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p if request.top_p > 0 else 1.0,
        seed=request.seed or None,
    )


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    """Translates gRPC messages into engine calls.

    Requests run on a thread pool, one call per decode. Whether those decodes
    share forward passes is the engine's business, not this class's: the
    `batched` backend merges them through the scheduler in `batching.py`,
    while `reference`, `paged` and `speculative` decode one sequence at a
    time. Selecting between them is a deployment change, not a protocol one.
    """

    def __init__(self, handle: EngineHandle) -> None:
        self._handle = handle
        self._in_flight = 0
        self._lock = threading.Lock()
        metrics.track_servicer(self)

    def _validate(self, request, context) -> None:
        """Reject a malformed request. Shared by unary and streaming."""
        if request.max_tokens <= 0:
            # Counted with no latency: a rejected request never reached the
            # model, so it belongs in the totals but not in the histogram.
            metrics.record_request(outcome="rejected")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "max_tokens must be positive"
            )

        cap = self._handle.config.max_tokens_cap
        if request.max_tokens > cap:
            metrics.record_request(outcome="rejected")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"max_tokens {request.max_tokens} exceeds cap {cap}",
            )

    def _enter(self) -> None:
        with self._lock:
            self._in_flight += 1

    def _exit(self) -> None:
        with self._lock:
            self._in_flight -= 1

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def Generate(self, request, context):  # noqa: N802 - gRPC naming
        request_id = request.request_id or str(uuid.uuid4())

        self._validate(request, context)
        params = _params_from(request)

        self._enter()
        start = time.perf_counter()
        try:
            result = self._handle.engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                params=params,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            metrics.record_request(
                latency_seconds=time.perf_counter() - start, outcome="error"
            )
            logger.exception("generation failed request_id=%s", request_id)
            context.abort(grpc.StatusCode.INTERNAL, f"generation failed: {exc}")
        finally:
            self._exit()

        metrics.record_request(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_seconds=result.latency_ms / 1000.0,
        )

        logger.info(
            "request_id=%s prompt_tokens=%d completion_tokens=%d latency_ms=%.1f",
            request_id,
            result.prompt_tokens,
            result.completion_tokens,
            result.latency_ms,
        )

        return inference_pb2.GenerateResponse(
            text=result.text,
            finished=True,
            request_id=request_id,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
        )

    def GenerateStream(self, request, context):  # noqa: N802 - gRPC naming
        """Emit one message per token as it is decoded.

        Each message carries only the new token's text, so a client renders by
        appending rather than by diffing against what it already showed. Token
        counts are reported on the final message, where they are known.
        """
        request_id = request.request_id or str(uuid.uuid4())
        self._validate(request, context)

        engine = self._handle.engine
        if not hasattr(engine, "iter_generate"):
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"backend {self._handle.backend!r} cannot stream; "
                "use the paged, batched or speculative backend",
            )

        params = _params_from(request)
        completion_tokens = 0
        prompt_tokens = 0

        self._enter()
        start = time.perf_counter()
        try:
            for chunk in engine.iter_generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                params=params,
            ):
                if chunk.token_id is not None:
                    completion_tokens += 1
                prompt_tokens = chunk.prompt_tokens
                yield inference_pb2.GenerateResponse(
                    text=chunk.text,
                    finished=chunk.finished,
                    request_id=request_id,
                    prompt_tokens=chunk.prompt_tokens,
                    completion_tokens=completion_tokens,
                    finish_reason=chunk.finish_reason or "",
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            metrics.record_request(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=time.perf_counter() - start,
                outcome="error",
            )
            logger.exception("stream failed request_id=%s", request_id)
            context.abort(grpc.StatusCode.INTERNAL, f"generation failed: {exc}")
        else:
            # `else`, not `finally`: a client that walks away mid-stream closes
            # this generator with GeneratorExit, and that is neither a served
            # request nor an engine error, so it is counted as neither.
            metrics.record_request(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=time.perf_counter() - start,
            )
        finally:
            self._exit()

    def Health(self, request, context):  # noqa: N802 - gRPC naming
        return inference_pb2.HealthResponse(
            ready=self._handle.ready,
            model=self._handle.config.model_name,
            device=self._handle.config.device,
            in_flight=self.in_flight,
        )


def serve(config: WorkerConfig | None = None) -> None:
    """Start the gRPC server and block until terminated."""
    logging.basicConfig(
        level=os.getenv("TESSERA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = config or WorkerConfig.from_env()

    handle = EngineHandle(config)
    handle.load()

    # Production runs gRPC, so the scrape target has to exist here rather than
    # only on the FastAPI dev wrapper.
    metrics.start_metrics_server(config.metrics_port)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(os.getenv("TESSERA_THREADS", "8"))),
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )
    inference_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer(handle), server
    )
    bind = f"0.0.0.0:{config.grpc_port}"
    server.add_insecure_port(bind)
    server.start()
    logger.info("worker listening on %s", bind)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()

    logger.info("shutting down")
    server.stop(grace=5).wait()

    # The batched backend owns a scheduler thread. Stopping the gRPC server
    # does not stop it, so ask the engine to close if it knows how.
    close = getattr(handle.engine if handle.ready else None, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    serve()
