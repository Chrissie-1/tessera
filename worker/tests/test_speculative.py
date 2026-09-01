"""Phase 3: speculative decoding.

Speculation is a throughput trick that must not move a single token. The
drafter is varied deliberately across these tests -- perfect, useless, and
partly right -- and the output is required to be identical every time. If
draft quality could change the answer, the verification step would be wrong.
"""

from __future__ import annotations

import pytest
import torch

from tessera_worker.sampling import SamplingParams
from tessera_worker.speculative import ModelDrafter, SpeculativeEngine


class WrongDrafter:
    """Proposes a token that is almost never the target's choice.

    Every proposal should be rejected, so the engine falls back to one token
    per pass and must still produce the reference output.
    """

    def __init__(self, token: int = 0) -> None:
        self.token = token
        self.calls = 0

    def propose(self, context_ids, k):
        self.calls += 1
        return [self.token] * k


class AlternatingDrafter:
    """Right half the time, to exercise partial acceptance and truncation."""

    def __init__(self, engine) -> None:
        self.inner = ModelDrafter(engine)

    def propose(self, context_ids, k):
        proposed = self.inner.propose(context_ids, k)
        # Corrupt every second proposal so some are accepted and some are not.
        return [t if i % 2 == 0 else 0 for i, t in enumerate(proposed)]


@pytest.fixture(scope="module")
def speculative(config):
    return SpeculativeEngine(config, lookahead=4, num_blocks=256, block_size=8)


PROMPTS = [
    "The capital of France is",
    "Hello world",
    "",
    "Numbers: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
]


# -- configuration ----------------------------------------------------------


def test_rejects_non_positive_lookahead(config):
    with pytest.raises(ValueError):
        SpeculativeEngine(config, lookahead=0)


def test_acceptance_rate_is_zero_before_any_proposal(config):
    engine = SpeculativeEngine(config, lookahead=2, num_blocks=32, block_size=8)

    assert engine.acceptance_rate == 0.0


# -- exactness --------------------------------------------------------------


@pytest.mark.parametrize("prompt", PROMPTS, ids=["capital", "hello", "empty", "long"])
def test_matches_reference_with_a_perfect_drafter(engine, speculative, prompt):
    """The default drafter is the target itself, so nothing should be rejected."""
    expected = engine.generate(prompt, max_tokens=10)
    actual = speculative.generate(prompt, max_tokens=10)

    assert actual.token_ids == expected.token_ids
    assert actual.text == expected.text


@pytest.mark.parametrize("prompt", PROMPTS, ids=["capital", "hello", "empty", "long"])
def test_matches_reference_with_a_useless_drafter(engine, config, prompt):
    """A drafter that is always wrong may cost speed, never correctness."""
    drafter = WrongDrafter()
    spec = SpeculativeEngine(
        config, drafter=drafter, lookahead=4, num_blocks=256, block_size=8
    )

    expected = engine.generate(prompt, max_tokens=10)
    actual = spec.generate(prompt, max_tokens=10)

    assert actual.token_ids == expected.token_ids
    assert drafter.calls > 0


def test_matches_reference_with_a_partly_right_drafter(engine, config):
    """Partial acceptance is where the cache truncation has to be exact."""
    spec = SpeculativeEngine(config, lookahead=4, num_blocks=256, block_size=8)
    spec.drafter = AlternatingDrafter(spec)

    expected = engine.generate("The capital of France is", max_tokens=12)
    actual = spec.generate("The capital of France is", max_tokens=12)

    assert actual.token_ids == expected.token_ids


@pytest.mark.parametrize("lookahead", [1, 2, 3, 8])
def test_output_is_independent_of_lookahead(engine, config, lookahead):
    spec = SpeculativeEngine(config, lookahead=lookahead, num_blocks=256, block_size=8)

    expected = engine.generate("Hello world", max_tokens=9)
    actual = spec.generate("Hello world", max_tokens=9)

    assert actual.token_ids == expected.token_ids


@pytest.mark.parametrize("max_tokens", [1, 2, 5, 11])
def test_token_budget_is_exact(engine, speculative, max_tokens):
    expected = engine.generate("The capital of France is", max_tokens=max_tokens)
    actual = speculative.generate("The capital of France is", max_tokens=max_tokens)

    assert actual.completion_tokens == expected.completion_tokens
    assert actual.token_ids == expected.token_ids


def test_accounting_matches_reference(engine, speculative):
    expected = engine.generate("Hello world", max_tokens=7)
    actual = speculative.generate("Hello world", max_tokens=7)

    assert actual.prompt_tokens == expected.prompt_tokens
    assert actual.finish_reason == expected.finish_reason


# -- streaming --------------------------------------------------------------


def test_streamed_deltas_match_the_completion(speculative):
    chunks = list(speculative.iter_generate("Hello world", max_tokens=6))
    streamed = "".join(c.text for c in chunks)

    assert streamed == speculative.generate("Hello world", max_tokens=6).text
    assert chunks[-1].finished is True


def test_only_the_last_chunk_is_final(speculative):
    chunks = list(speculative.iter_generate("Hello world", max_tokens=5))

    assert all(not c.finished for c in chunks[:-1])


# -- speculative sampling ----------------------------------------------------


def test_sampling_falls_back_without_proposal_probabilities(config, engine):
    """A drafter that cannot report q leaves the acceptance ratio undefined."""
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=5)
    spec = SpeculativeEngine(
        config, drafter=WrongDrafter(), lookahead=4, num_blocks=256, block_size=8
    )

    expected = engine.generate("Once upon a time", max_tokens=6, params=params)
    actual = spec.generate("Once upon a time", max_tokens=6, params=params)

    assert actual.token_ids == expected.token_ids


def test_sampling_speculates_when_the_drafter_reports_probabilities(config):
    """The default drafter can report q, so sampling should not fall back."""
    params = SamplingParams(temperature=1.0, top_p=1.0, seed=5)
    spec = SpeculativeEngine(config, lookahead=4, num_blocks=256, block_size=8)
    spec.generate("Once upon a time", max_tokens=8, params=params)

    assert spec.proposed_tokens > 0


def test_sampling_is_reproducible_under_a_seed(config):
    params = SamplingParams(temperature=1.0, top_p=0.9, seed=99)

    first = SpeculativeEngine(config, lookahead=3, num_blocks=256, block_size=8)
    second = SpeculativeEngine(config, lookahead=3, num_blocks=256, block_size=8)

    assert (
        first.generate("Hello world", max_tokens=6, params=params).token_ids
        == second.generate("Hello world", max_tokens=6, params=params).token_ids
    )


def test_accepted_tokens_never_exceed_proposed(speculative):
    speculative.generate("Hello world", max_tokens=8)

    assert speculative.accepted_tokens <= speculative.proposed_tokens


def test_judge_reproduces_the_target_distribution(config):
    """The guarantee: accept-or-residual composes to exactly the target p.

    Driven with synthetic distributions rather than the model, so the only
    thing under test is the acceptance rule. A drafter deliberately skewed
    away from the target makes rejection the common path.
    """
    spec = SpeculativeEngine(config, lookahead=1, num_blocks=32, block_size=8)

    target = torch.tensor([0.5, 0.3, 0.15, 0.05])
    draft = torch.tensor([0.1, 0.2, 0.3, 0.4])
    # softmax(log p) == p, so the engine sees exactly `target`.
    logits = torch.log(target)
    params = SamplingParams(temperature=1.0, top_p=1.0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)

    trials = 20000
    counts = torch.zeros(4)
    proposals = torch.multinomial(
        draft, num_samples=trials, replacement=True, generator=generator
    )
    for proposal in proposals.tolist():
        keep, replacement = spec._judge(logits, proposal, draft, params, generator)
        counts[proposal if keep else replacement] += 1

    empirical = counts / trials
    assert torch.allclose(empirical, target, atol=0.02), empirical


def test_judge_accepts_everything_when_draft_matches_target(config):
    """p == q makes the ratio 1, so nothing should ever be rejected."""
    spec = SpeculativeEngine(config, lookahead=1, num_blocks=32, block_size=8)

    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    params = SamplingParams(temperature=1.0, top_p=1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)

    for token in range(4):
        keep, _ = spec._judge(torch.log(probs), token, probs, params, generator)
        assert keep


def test_judge_rejects_a_token_the_draft_could_not_propose(config):
    """q(x) == 0 makes the ratio unbounded; it must not be accepted."""
    spec = SpeculativeEngine(config, lookahead=1, num_blocks=32, block_size=8)

    target = torch.tensor([0.25, 0.25, 0.25, 0.25])
    draft = torch.tensor([0.5, 0.5, 0.0, 0.0])
    params = SamplingParams(temperature=1.0, top_p=1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)

    keep, replacement = spec._judge(torch.log(target), 2, draft, params, generator)

    assert keep is False
    # The residual has mass only where the draft under-covered the target.
    assert replacement in {2, 3}


# -- cache hygiene ----------------------------------------------------------


def test_blocks_are_returned_after_a_request(speculative):
    before = speculative.cache.allocator.free_blocks
    speculative.generate("The capital of France is", max_tokens=8)

    assert speculative.cache.allocator.free_blocks == before


def test_rejected_proposals_do_not_leak_blocks(config):
    """Truncation has to give rejected proposals' blocks back."""
    spec = SpeculativeEngine(
        config, drafter=WrongDrafter(), lookahead=8, num_blocks=256, block_size=4
    )
    before = spec.cache.allocator.free_blocks
    spec.generate("The capital of France is", max_tokens=10)

    assert spec.cache.allocator.free_blocks == before


def test_acceptance_rate_is_high_for_a_perfect_drafter(config):
    spec = SpeculativeEngine(config, lookahead=4, num_blocks=256, block_size=8)
    spec.generate("The capital of France is", max_tokens=12)

    assert spec.acceptance_rate == pytest.approx(1.0)


def test_acceptance_rate_is_low_for_a_useless_drafter(config):
    spec = SpeculativeEngine(
        config, drafter=WrongDrafter(), lookahead=4, num_blocks=256, block_size=8
    )
    spec.generate("The capital of France is", max_tokens=12)

    assert spec.acceptance_rate < 0.5
