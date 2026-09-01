"""gRPC server exposing the inference engine to the Rust gateway."""

from __future__ import annotations

import logging
import os
import signal
import threading
import uuid
from concurrent import futures

import grpc

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

    Requests run on a thread pool, one decode per call. The continuous
    batching scheduler in `batching.py` is what shares a single forward pass
    across concurrent sequences; wiring it in behind this interface is a
    deployment change, not a protocol one.
    """

    def __init__(self, handle: EngineHandle) -> None:
        self._handle = handle
        self._in_flight = 0
        self._lock = threading.Lock()

    def _validate(self, request, context) -> None:
        """Reject a malformed request. Shared by unary and streaming."""
        if request.max_tokens <= 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "max_tokens must be positive"
            )

        cap = self._handle.config.max_tokens_cap
        if request.max_tokens > cap:
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
        try:
            result = self._handle.engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                params=params,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            logger.exception("generation failed request_id=%s", request_id)
            context.abort(grpc.StatusCode.INTERNAL, f"generation failed: {exc}")
        finally:
            self._exit()

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
                "use the paged or speculative backend",
            )

        params = _params_from(request)
        completion_tokens = 0

        self._enter()
        try:
            for chunk in engine.iter_generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                params=params,
            ):
                if chunk.token_id is not None:
                    completion_tokens += 1
                yield inference_pb2.GenerateResponse(
                    text=chunk.text,
                    finished=chunk.finished,
                    request_id=request_id,
                    prompt_tokens=chunk.prompt_tokens,
                    completion_tokens=completion_tokens,
                    finish_reason=chunk.finish_reason or "",
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            logger.exception("stream failed request_id=%s", request_id)
            context.abort(grpc.StatusCode.INTERNAL, f"generation failed: {exc}")
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


if __name__ == "__main__":
    serve()
