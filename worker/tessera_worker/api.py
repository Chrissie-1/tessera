"""FastAPI wrapper so the worker can be exercised without the Rust gateway.

This is a development and debugging surface. Production traffic goes
gateway -> gRPC -> worker; this module skips the gateway entirely.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import metrics
from .config import WorkerConfig
from .engine import EngineHandle
from .sampling import SamplingParams

logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=16, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False


class CompletionResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_ms: float


def create_app(config: WorkerConfig | None = None) -> FastAPI:
    config = config or WorkerConfig.from_env()
    handle = EngineHandle(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load before serving so a broken model fails startup, not the first
        # request -- same contract as EngineHandle's explicit load().
        handle.load()
        yield

    app = FastAPI(title="Tessera Worker", version="0.1.0", lifespan=lifespan)

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        """Scrape target for the dev wrapper.

        The gRPC worker exports the same registry over its own port (see
        `TESSERA_METRICS_PORT`); this endpoint exists so the FastAPI surface
        is not a blind spot when someone is debugging through it.
        """
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)

    @app.get("/health")
    def health() -> dict:
        return {
            "ready": handle.ready,
            "model": config.model_name,
            "device": config.device,
            "backend": handle.backend,
        }

    def _sse(request: CompletionRequest, params: SamplingParams) -> StreamingResponse:
        """Stream tokens as server-sent events, matching the gateway's shape.

        Each event carries one token's delta, so a client appends rather than
        diffing, and the stream is terminated by `[DONE]`.
        """
        engine = handle.engine

        def events():
            start = time.perf_counter()
            prompt_tokens = 0
            completion_tokens = 0
            try:
                for chunk in engine.iter_generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    params=params,
                ):
                    prompt_tokens = chunk.prompt_tokens
                    if chunk.token_id is not None:
                        completion_tokens += 1
                    payload = {
                        "object": "text_completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "text": chunk.text,
                                "finish_reason": chunk.finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:  # noqa: BLE001 - relayed to the client
                # Headers are already sent, so the only way to report a
                # mid-stream failure is inside the stream itself.
                metrics.record_request(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_seconds=time.perf_counter() - start,
                    outcome="error",
                )
                logger.exception("stream failed")
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
            else:
                # `else`, not `finally`: a client that disconnects mid-stream
                # closes this generator with GeneratorExit, which is neither a
                # served request nor an engine error, so it counts as neither.
                metrics.record_request(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_seconds=time.perf_counter() - start,
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(request: CompletionRequest):
        if request.max_tokens > config.max_tokens_cap:
            metrics.record_request(outcome="rejected")
            raise HTTPException(
                status_code=422,
                detail=f"max_tokens exceeds cap {config.max_tokens_cap}",
            )
        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
        )

        if request.stream:
            if not hasattr(handle.engine, "iter_generate"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"backend {handle.backend!r} cannot stream; use the "
                        "paged, batched or speculative backend"
                    ),
                )
            return _sse(request, params)

        start = time.perf_counter()
        try:
            result = handle.engine.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                params=params,
            )
        except Exception:
            metrics.record_request(
                latency_seconds=time.perf_counter() - start, outcome="error"
            )
            raise

        metrics.record_request(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_seconds=result.latency_ms / 1000.0,
        )
        return CompletionResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
            latency_ms=result.latency_ms,
        )

    return app


app = create_app()
