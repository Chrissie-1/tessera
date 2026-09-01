"""FastAPI wrapper so the worker can be exercised without the Rust gateway.

This is a development and debugging surface. Production traffic goes
gateway -> gRPC -> worker; this module skips the gateway entirely.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

    @app.get("/health")
    def health() -> dict:
        return {
            "ready": handle.ready,
            "model": config.model_name,
            "device": config.device,
            "backend": handle.backend,
        }

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(request: CompletionRequest) -> CompletionResponse:
        if request.max_tokens > config.max_tokens_cap:
            raise HTTPException(
                status_code=422,
                detail=f"max_tokens exceeds cap {config.max_tokens_cap}",
            )
        result = handle.engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            params=SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            ),
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
