"""Token sampling primitives shared by every decoding path.

Phase 2 (paged) and Phase 3 (speculative) reuse these exact functions, so that
a difference in output between engines can never be blamed on a difference in
how the next token was drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """temperature <= 0 selects greedy decoding (argmax)."""

    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out the tail of the distribution beyond cumulative mass `top_p`.

    Args:
        probs: (..., vocab) normalised probabilities.
        top_p: nucleus threshold in (0, 1]. Values >= 1 are a no-op.

    Returns:
        Renormalised probabilities with the same shape.
    """
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Keep every token up to and including the one that crosses the threshold,
    # so the nucleus is never empty even when one token holds > top_p mass.
    keep = cumulative - sorted_probs < top_p
    keep[..., 0] = True

    sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


def logits_to_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Convert raw logits into the sampling distribution (temperature + top-p).

    Args:
        logits: (..., vocab) raw model outputs.
        params: sampling configuration; must not be greedy.
    """
    if params.greedy:
        raise ValueError("logits_to_probs is undefined for greedy sampling")
    probs = torch.softmax(logits.float() / params.temperature, dim=-1)
    return apply_top_p(probs, params.top_p)


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    """Draw the next token id from a single position's logits.

    Args:
        logits: (vocab,) raw logits for the position being decoded.
        params: sampling configuration.
        generator: torch RNG, supplied when the caller needs reproducibility.
    """
    if params.greedy:
        return int(torch.argmax(logits, dim=-1).item())

    probs = logits_to_probs(logits, params)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def make_generator(device: str, seed: int | None) -> torch.Generator | None:
    """Build a seeded RNG on `device`, or None when the caller wants entropy."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
