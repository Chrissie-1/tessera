"""Shared fixtures. Tests default to a tiny random GPT-2 so CI needs no GPU."""

from __future__ import annotations

import os

import pytest
import torch

from tessera_worker.config import TEST_MODEL, WorkerConfig


@pytest.fixture(scope="session")
def config() -> WorkerConfig:
    return WorkerConfig(
        model_name=os.getenv("TEST_MODEL", TEST_MODEL),
        device="cpu",
        dtype=torch.float32,
        grpc_port=50551,
        http_port=8801,
        max_tokens_cap=64,
    )


@pytest.fixture(scope="session")
def engine(config: WorkerConfig):
    from tessera_worker.model import ReferenceEngine

    return ReferenceEngine(config)
