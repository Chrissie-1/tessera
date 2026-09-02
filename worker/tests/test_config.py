"""Configuration: architecture resolution and environment overrides.

`num_layers` and `max_positions` used to be read straight off GPT-2's own
attribute names (`n_layer`, `n_positions`), which meant the worker loaded
exactly one model family and raised `AttributeError` on every other. These
tests pin the resolution down across architectures without downloading any
weights -- a config object is all the code under test ever looks at.
"""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, LlamaConfig, MistralConfig

from tessera_worker.config import (
    DEFAULT_LOOKAHEAD,
    DEFAULT_MAX_BATCH_SIZE,
    WorkerConfig,
    max_positions,
    num_layers,
)
from tessera_worker.paged import DEFAULT_BLOCK_SIZE, DEFAULT_NUM_BLOCKS

# -- architecture resolution -------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (GPT2Config(n_layer=12), 12),
        (LlamaConfig(num_hidden_layers=32), 32),
        (MistralConfig(num_hidden_layers=48), 48),
    ],
    ids=["gpt2", "llama", "mistral"],
)
def test_num_layers_resolves_across_architectures(config, expected):
    assert num_layers(config) == expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (GPT2Config(n_positions=1024), 1024),
        (LlamaConfig(max_position_embeddings=4096), 4096),
        (MistralConfig(max_position_embeddings=32768), 32768),
    ],
    ids=["gpt2", "llama", "mistral"],
)
def test_max_positions_resolves_across_architectures(config, expected):
    assert max_positions(config) == expected


def test_num_layers_rejects_a_config_it_cannot_read():
    """Better to fail loudly at load than to silently size the cache wrong."""

    class Opaque:
        pass

    with pytest.raises(AttributeError, match="cannot determine layer count"):
        num_layers(Opaque())


def test_max_positions_falls_back_when_unbounded():
    """RoPE models often declare no limit; the decode loop still needs one."""

    class Opaque:
        pass

    assert max_positions(Opaque(), default=2048) == 2048


# -- environment overrides ---------------------------------------------------


def test_sizing_defaults_match_the_engine_constants():
    config = WorkerConfig.from_env()

    assert config.max_batch_size == DEFAULT_MAX_BATCH_SIZE
    assert config.num_blocks == DEFAULT_NUM_BLOCKS
    assert config.block_size == DEFAULT_BLOCK_SIZE
    assert config.lookahead == DEFAULT_LOOKAHEAD


def test_sizing_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("TESSERA_MAX_BATCH_SIZE", "3")
    monkeypatch.setenv("TESSERA_NUM_BLOCKS", "64")
    monkeypatch.setenv("TESSERA_BLOCK_SIZE", "8")
    monkeypatch.setenv("TESSERA_LOOKAHEAD", "2")

    config = WorkerConfig.from_env()

    assert (config.max_batch_size, config.num_blocks) == (3, 64)
    assert (config.block_size, config.lookahead) == (8, 2)


def test_metrics_port_is_off_unless_the_environment_sets_it(monkeypatch):
    """Nothing binds a metrics port by accident; it is opt-in."""
    monkeypatch.delenv("TESSERA_METRICS_PORT", raising=False)

    assert WorkerConfig.from_env().metrics_port == 0

    monkeypatch.setenv("TESSERA_METRICS_PORT", "9187")

    assert WorkerConfig.from_env().metrics_port == 9187


def test_engines_take_their_sizing_from_the_config(config):
    """A configured value must reach the engine without an explicit argument."""
    from tessera_worker.paged_engine import PagedEngine

    sized = WorkerConfig(
        model_name=config.model_name,
        device="cpu",
        dtype=torch.float32,
        grpc_port=0,
        http_port=0,
        max_tokens_cap=16,
        num_blocks=48,
        block_size=8,
    )
    engine = PagedEngine(sized)

    assert engine.cache.allocator.num_blocks == 48
    assert engine.cache.block_size == 8


def test_an_explicit_argument_still_wins_over_the_config(config):
    from tessera_worker.paged_engine import PagedEngine

    sized = WorkerConfig(
        model_name=config.model_name,
        device="cpu",
        dtype=torch.float32,
        grpc_port=0,
        http_port=0,
        max_tokens_cap=16,
        num_blocks=48,
        block_size=8,
    )
    engine = PagedEngine(sized, num_blocks=16)

    assert engine.cache.allocator.num_blocks == 16
    assert engine.cache.block_size == 8
