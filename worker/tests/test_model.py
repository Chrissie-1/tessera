"""Reference engine behaviour.

These assertions define what "correct output" means for the whole project:
Phase 2's paged cache and Phase 3's speculative decoder are tested against
this engine, not against each other.
"""

from __future__ import annotations

import pytest
import torch

from tessera_worker.sampling import SamplingParams


def test_generate_respects_max_tokens(engine):
    result = engine.generate("The capital of France is", max_tokens=5)
    assert result.completion_tokens <= 5
    assert len(result.token_ids) == result.completion_tokens


def test_greedy_decoding_is_deterministic(engine):
    first = engine.generate("Hello world", max_tokens=8)
    second = engine.generate("Hello world", max_tokens=8)
    assert first.token_ids == second.token_ids
    assert first.text == second.text


def test_prompt_tokens_match_tokenizer(engine):
    prompt = "The quick brown fox jumps over the lazy dog"
    result = engine.generate(prompt, max_tokens=2)
    assert result.prompt_tokens == len(engine.tokenizer.encode(prompt))


def test_empty_prompt_does_not_crash(engine):
    result = engine.generate("", max_tokens=3)
    assert result.prompt_tokens >= 1
    assert isinstance(result.text, str)


def test_seeded_sampling_is_reproducible(engine):
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=42)
    first = engine.generate("Once upon a time", max_tokens=6, params=params)
    second = engine.generate("Once upon a time", max_tokens=6, params=params)
    assert first.token_ids == second.token_ids


def test_forward_logits_shape_matches_vocab(engine):
    token_ids = engine.tokenizer.encode("Testing logits")
    logits = engine.forward_logits(token_ids)
    assert logits.shape == (engine.model.config.vocab_size,)
    assert torch.isfinite(logits).all()


def test_forward_logits_is_stateless(engine):
    """A dense forward must not depend on prior calls; there is no cache."""
    token_ids = engine.tokenizer.encode("Statelessness check")
    first = engine.forward_logits(token_ids)
    engine.forward_logits(engine.tokenizer.encode("An unrelated sequence"))
    second = engine.forward_logits(token_ids)
    assert torch.equal(first, second)


def test_generation_never_exceeds_context_window(engine):
    limit = engine.max_position_embeddings
    long_prompt = "word " * max(1, limit - 2)
    result = engine.generate(long_prompt, max_tokens=32)
    assert result.prompt_tokens + result.completion_tokens <= limit


@pytest.mark.parametrize("max_tokens", [1, 2, 4])
def test_token_count_is_exact_without_eos(engine, max_tokens):
    result = engine.generate("Numbers: 1 2 3", max_tokens=max_tokens)
    if result.finish_reason == "length":
        assert result.completion_tokens == max_tokens
