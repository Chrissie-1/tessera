"""Sampling primitives. These are the shared contract for every decode path."""

from __future__ import annotations

import torch

from tessera_worker.sampling import (
    SamplingParams,
    apply_top_p,
    logits_to_probs,
    make_generator,
    residual_probs,
    sample_token,
)


def test_greedy_is_argmax():
    logits = torch.tensor([0.1, 5.0, -3.0, 2.0])
    assert sample_token(logits, SamplingParams(temperature=0.0)) == 1


def test_top_p_keeps_nucleus_and_renormalises():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    filtered = apply_top_p(probs, top_p=0.8)

    assert torch.isclose(filtered.sum(), torch.tensor(1.0))
    # 0.5 + 0.3 reaches the threshold, so the tail is dropped.
    assert filtered[2] == 0.0
    assert filtered[3] == 0.0


def test_top_p_never_empties_the_nucleus():
    # One token holds more mass than top_p; it must still survive.
    probs = torch.tensor([0.9, 0.07, 0.03])
    filtered = apply_top_p(probs, top_p=0.5)

    assert filtered[0] == 1.0
    assert torch.isclose(filtered.sum(), torch.tensor(1.0))


def test_top_p_above_one_is_a_noop():
    probs = torch.tensor([0.5, 0.3, 0.2])
    assert torch.equal(apply_top_p(probs, top_p=1.0), probs)


def test_temperature_flattens_distribution():
    logits = torch.tensor([1.0, 2.0, 3.0])
    cold = logits_to_probs(logits, SamplingParams(temperature=0.5))
    hot = logits_to_probs(logits, SamplingParams(temperature=2.0))

    # Lower temperature concentrates mass on the argmax.
    assert cold.max() > hot.max()


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(64)
    params = SamplingParams(temperature=1.0, seed=1234)

    first = sample_token(logits, params, make_generator("cpu", 1234))
    second = sample_token(logits, params, make_generator("cpu", 1234))
    assert first == second


def test_greedy_rejects_probability_conversion():
    import pytest

    with pytest.raises(ValueError):
        logits_to_probs(torch.randn(8), SamplingParams(temperature=0.0))


def test_residual_keeps_only_what_the_draft_under_covered():
    target = torch.tensor([0.5, 0.3, 0.2])
    draft = torch.tensor([0.1, 0.6, 0.3])

    residual = residual_probs(target, draft)

    # The draft over-covered tokens 1 and 2, so only token 0 has mass left.
    assert residual[0] == 1.0
    assert residual[1] == 0.0
    assert residual[2] == 0.0


def test_residual_is_normalised():
    target = torch.tensor([0.4, 0.4, 0.2])
    draft = torch.tensor([0.1, 0.1, 0.8])

    residual = residual_probs(target, draft)

    assert torch.isclose(residual.sum(), torch.tensor(1.0))
    assert (residual >= 0).all()


def test_residual_falls_back_to_the_target_when_fully_covered():
    """p == q leaves no residual mass; the fallback must still be a distribution."""
    probs = torch.tensor([0.6, 0.4])

    residual = residual_probs(probs, probs)

    assert torch.isclose(residual.sum(), torch.tensor(1.0))
    assert torch.equal(residual, probs)


def test_residual_ignores_draft_mass_beyond_the_target():
    target = torch.tensor([1.0, 0.0])
    draft = torch.tensor([0.5, 0.5])

    residual = residual_probs(target, draft)

    assert residual[0] == 1.0
    assert residual[1] == 0.0
