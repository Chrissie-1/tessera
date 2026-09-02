"""Runtime configuration, resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .paged import DEFAULT_BLOCK_SIZE, DEFAULT_NUM_BLOCKS

# A ~2 MB randomly-initialised GPT-2 used by the test suite. Real serving
# defaults to gpt2 (124M).
TEST_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_MODEL = "gpt2"

# Engine sizing defaults live here, next to the environment variables that
# override them, so the dataclass and the engines cannot drift apart.
DEFAULT_MAX_BATCH_SIZE = 8
DEFAULT_LOOKAHEAD = 4


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


def num_layers(model_config) -> int:
    """Decoder layer count, whatever the architecture calls it.

    `num_hidden_layers` is the name every modern `PretrainedConfig` answers
    to -- GPT-2 included, since `GPT2Config.attribute_map` aliases it onto
    `n_layer`. Reading `n_layer` directly, as this used to, worked on GPT-2
    and raised `AttributeError` on everything else.
    """
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model_config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(
        f"cannot determine layer count for config {type(model_config).__name__}"
    )


def max_positions(model_config, default: int = 1024) -> int:
    """Longest sequence the model has positions for.

    Falls back to `default` for architectures with no fixed limit (RoPE
    models often leave it unset), because the decode loop needs *some* bound
    to size its token budget against.
    """
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(model_config, attr, None)
        if value is not None:
            return int(value)
    return default


@dataclass(frozen=True)
class WorkerConfig:
    model_name: str
    device: str
    dtype: torch.dtype
    grpc_port: int
    http_port: int
    max_tokens_cap: int
    # 0 disables the Prometheus HTTP listener. Off by default because the
    # gRPC server is also run embedded (tests, bench harness), and those must
    # not silently bind a port.
    metrics_port: int = 0
    # Engine sizing. Defaults mirror the constructor defaults of the engines
    # themselves, so setting nothing behaves exactly as it did before these
    # became configurable.
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    num_blocks: int = DEFAULT_NUM_BLOCKS
    block_size: int = DEFAULT_BLOCK_SIZE
    lookahead: int = DEFAULT_LOOKAHEAD

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
            metrics_port=int(os.getenv("TESSERA_METRICS_PORT", "0")),
            max_batch_size=int(
                os.getenv("TESSERA_MAX_BATCH_SIZE", str(DEFAULT_MAX_BATCH_SIZE))
            ),
            num_blocks=int(os.getenv("TESSERA_NUM_BLOCKS", str(DEFAULT_NUM_BLOCKS))),
            block_size=int(os.getenv("TESSERA_BLOCK_SIZE", str(DEFAULT_BLOCK_SIZE))),
            lookahead=int(os.getenv("TESSERA_LOOKAHEAD", str(DEFAULT_LOOKAHEAD))),
        )
