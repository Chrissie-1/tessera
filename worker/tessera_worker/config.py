"""Runtime configuration, resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# A ~2 MB randomly-initialised GPT-2 used by the test suite. Real serving
# defaults to gpt2 (124M).
TEST_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_MODEL = "gpt2"


def _resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(raw: str, device: str) -> torch.dtype:
    if raw != "auto":
        return getattr(torch, raw)
    # float32 everywhere by default: the paged-vs-dense equivalence tests in
    # Phase 2 compare logits, and fp16 accumulation differences would make
    # "byte-identical" an unreachable bar. Opt into fp16 explicitly via
    # TESSERA_DTYPE=float16 when benchmarking.
    return torch.float32


@dataclass(frozen=True)
class WorkerConfig:
    model_name: str
    device: str
    dtype: torch.dtype
    grpc_port: int
    http_port: int
    max_tokens_cap: int

    @classmethod
    def from_env(cls) -> WorkerConfig:
        device = _resolve_device(os.getenv("TESSERA_DEVICE", "auto"))
        return cls(
            model_name=os.getenv("TESSERA_MODEL", DEFAULT_MODEL),
            device=device,
            dtype=_resolve_dtype(os.getenv("TESSERA_DTYPE", "auto"), device),
            grpc_port=int(os.getenv("TESSERA_GRPC_PORT", "50051")),
            http_port=int(os.getenv("TESSERA_HTTP_PORT", "8000")),
            max_tokens_cap=int(os.getenv("TESSERA_MAX_TOKENS_CAP", "512")),
        )
