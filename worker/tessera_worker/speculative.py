"""Phase 3: speculative decoding.

Decoding is memory-bound: a forward pass over one token costs almost the same
as a forward pass over several, because the time goes into moving weights, not
into the arithmetic. Speculative decoding spends that slack. A cheap drafter
proposes k tokens, the target model checks all k in a single pass, and every
proposal the target would have produced anyway is kept.

The guarantee that makes this worth doing is that it is not an approximation.
Under greedy decoding the accepted tokens are exactly the tokens the target
would have emitted alone -- a proposal is kept only when it equals the target's
own argmax -- so a bad drafter costs throughput and never accuracy. The tests
enforce that by running a deliberately wrong drafter and still demanding output
identical to the reference engine.

Sampling is exact too, by a different argument. A proposal drawn from the
drafter's q is accepted with probability min(1, p/q), and a rejection is
resolved by drawing from the normalised residual max(0, p - q). Those two steps
compose to exactly p, so the tokens are distributed as though the target model
had sampled them alone. That requires the drafter to report the distribution it
sampled from; a drafter that cannot falls back to ordinary paged decoding
rather than quietly sampling from the wrong distribution.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from typing import Protocol

import torch

from .config import WorkerConfig
from .paged import DEFAULT_BLOCK_SIZE, DEFAULT_NUM_BLOCKS
from .paged_engine import PagedEngine, StreamChunk
from .sampling import (
    SamplingParams,
    logits_to_probs,
    make_generator,
    residual_probs,
    sample_token,
)

logger = logging.getLogger(__name__)

DEFAULT_LOOKAHEAD = 4


class Drafter(Protocol):
    """Proposes continuations. Quality affects speed, never correctness."""

    def propose(self, context_ids: list[int], k: int) -> list[int]:
        """Return up to `k` proposed token ids following `context_ids`."""


class SamplingDrafter(Drafter, Protocol):
    """A drafter that can also report what it sampled from.

    Speculative *sampling* needs the proposal distribution, not just the
    proposal: the acceptance test is a ratio against it. A drafter that cannot
    supply one can still be used for greedy decoding.
    """

    def propose_with_probs(
        self,
        context_ids: list[int],
        k: int,
        params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Return proposals and the (vocab,) distribution each was drawn from.

        `generator` is threaded through so a seeded request stays reproducible
        across the draft as well as the verification.
        """


class ModelDrafter:
    """Greedy drafter backed by a small causal LM.

    Runs without a cache: a draft is short and thrown away often, so the
    bookkeeping to page it would cost more than the forward passes it saves.
    """

    def __init__(self, engine: PagedEngine) -> None:
        self.engine = engine

    @torch.inference_mode()
    def propose(self, context_ids: list[int], k: int) -> list[int]:
        proposed: list[int] = []
        context = list(context_ids)
        for _ in range(k):
            logits = self.engine.forward_logits(context)
            token = int(torch.argmax(logits, dim=-1).item())
            proposed.append(token)
            context.append(token)
        return proposed

    @torch.inference_mode()
    def propose_with_probs(
        self,
        context_ids: list[int],
        k: int,
        params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Draw `k` proposals from the drafter's own sampling distribution.

        The returned distribution is the one actually sampled from -- after
        temperature and top-p -- because that is the q the acceptance ratio is
        defined against. Reporting the raw softmax instead would silently break
        the distributional guarantee.
        """
        proposed: list[int] = []
        distributions: list[torch.Tensor] = []
        context = list(context_ids)
        for _ in range(k):
            logits = self.engine.forward_logits(context)
            probs = logits_to_probs(logits, params)
            token = int(
                torch.multinomial(probs, num_samples=1, generator=generator).item()
            )
            proposed.append(token)
            distributions.append(probs)
            context.append(token)
        return proposed, distributions


class SpeculativeEngine(PagedEngine):
    """Paged decoding with draft-and-verify on top.

    The target model's own paged cache holds only accepted tokens: rejected
    proposals are truncated away before the next round, so the cache never
    carries state for a token the model did not emit.
    """

    def __init__(
        self,
        config: WorkerConfig,
        drafter: Drafter | None = None,
        lookahead: int = DEFAULT_LOOKAHEAD,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__(config, num_blocks=num_blocks, block_size=block_size)
        if lookahead <= 0:
            raise ValueError("lookahead must be positive")
        self.lookahead = lookahead
        self.drafter = drafter if drafter is not None else ModelDrafter(self)

        # Throughput counters. Acceptance rate is the number that decides
        # whether speculation is paying for itself on a given workload.
        self.proposed_tokens = 0
        self.accepted_tokens = 0

    @property
    def acceptance_rate(self) -> float:
        if not self.proposed_tokens:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    @torch.inference_mode()
    def iter_generate(
        self,
        prompt: str,
        max_tokens: int,
        params: SamplingParams | None = None,
    ) -> Iterator[StreamChunk]:
        params = params or SamplingParams()
        if not params.greedy and not hasattr(self.drafter, "propose_with_probs"):
            # Without the proposal distribution the acceptance ratio is
            # undefined, so speculating would change what is sampled.
            yield from super().iter_generate(prompt, max_tokens, params)
            return

        prompt_ids: list[int] = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.eos_token_id or 0]
        prompt_tokens = len(prompt_ids)

        budget = min(max_tokens, self.max_position_embeddings - prompt_tokens)
        budget = max(0, budget)

        emitted: list[int] = []
        finish_reason = "length"
        generator = make_generator(self.config.device, params.seed)
        seq_id = self._new_sequence_id()
        self.cache.add_sequence(seq_id)

        try:
            device = self.config.device
            outputs = self.model(
                input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device),
                use_cache=True,
            )
            self.cache.append(seq_id, self.new_kv(outputs.past_key_values, 0))

            # `pending` is the emitted token whose keys and values are not in
            # the cache yet; it leads the next verification pass.
            pending = sample_token(outputs.logits[0, -1, :], params, generator)
            if pending == self.eos_token_id or budget == 0:
                finish_reason = "stop" if budget else "length"
                yield self._final(prompt_tokens, finish_reason)
                return

            while True:
                context = prompt_ids + emitted
                k = min(self.lookahead, budget - len(emitted) - 1)
                draft_probs: list[torch.Tensor] = []
                if k <= 0:
                    drafts = []
                elif params.greedy:
                    drafts = self.drafter.propose(context + [pending], k)
                else:
                    drafts, draft_probs = self.drafter.propose_with_probs(
                        context + [pending], k, params, generator
                    )
                cached_len = self.cache.length(seq_id)
                step_ids = [pending, *drafts]
                outputs = self.model(
                    input_ids=torch.tensor([step_ids], dtype=torch.long, device=device),
                    past_key_values=self.cache.to_cache(seq_id),
                    use_cache=True,
                )
                self.cache.append(
                    seq_id, self.new_kv(outputs.past_key_values, cached_len)
                )
                logits = outputs.logits[0]

                # Emit `pending`; it was already verified when it was chosen.
                stop = yield from self._emit(pending, emitted, prompt_tokens, budget)
                if stop is not None:
                    finish_reason = stop
                    return

                accepted = 0
                nxt: int | None = None
                for i, draft in enumerate(drafts):
                    # Counted at verification rather than at proposal time, so
                    # the rate reflects proposals actually judged -- a round cut
                    # short by the token budget skews neither counter.
                    self.proposed_tokens += 1
                    keep, replacement = self._judge(
                        logits[i],
                        draft,
                        draft_probs[i] if draft_probs else None,
                        params,
                        generator,
                    )
                    if not keep:
                        nxt = replacement
                        break
                    accepted += 1
                    self.accepted_tokens += 1
                    stop = yield from self._emit(draft, emitted, prompt_tokens, budget)
                    if stop is not None:
                        finish_reason = stop
                        return
                else:
                    # Every proposal held, so the pass also produced a token
                    # beyond them for free.
                    nxt = sample_token(logits[len(drafts)], params, generator)

                # Drop the keys and values of proposals that were rejected:
                # the pass wrote all of them, only the accepted prefix counts.
                self.cache.truncate(seq_id, cached_len + 1 + accepted)

                if nxt == self.eos_token_id:
                    finish_reason = "stop"
                    yield self._final(prompt_tokens, finish_reason)
                    return
                pending = nxt
        finally:
            self.cache.free_sequence(seq_id)

    def _judge(
        self,
        target_logits: torch.Tensor,
        draft: int,
        draft_probs: torch.Tensor | None,
        params: SamplingParams,
        generator: torch.Generator | None,
    ) -> tuple[bool, int]:
        """Decide whether to keep one proposal, and what to emit if not.

        Greedy keeps a proposal only when it is the target's argmax, which is
        a hard equality. Sampling keeps it with probability min(1, p/q) and
        resolves a rejection from the residual, which is what preserves the
        target distribution rather than merely approximating it.
        """
        if params.greedy:
            target = int(torch.argmax(target_logits, dim=-1).item())
            return target == draft, target

        p = logits_to_probs(target_logits, params)
        q = draft_probs
        # A token the draft could never have proposed cannot be accepted; the
        # ratio is unbounded, so treat it as an outright rejection.
        q_draft = float(q[draft]) if q is not None else 0.0
        if q_draft <= 0.0:
            return False, int(
                torch.multinomial(
                    residual_probs(p, q) if q is not None else p,
                    num_samples=1,
                    generator=generator,
                ).item()
            )

        ratio = float(p[draft]) / q_draft
        roll = torch.rand(1, generator=generator, device=p.device).item()
        if roll < min(1.0, ratio):
            return True, draft

        return False, int(
            torch.multinomial(
                residual_probs(p, q), num_samples=1, generator=generator
            ).item()
        )

    def _emit(
        self,
        token: int,
        emitted: list[int],
        prompt_tokens: int,
        budget: int,
    ) -> Generator[StreamChunk, None, str | None]:
        """Yield one token, returning a finish reason once the budget is spent."""
        emitted.append(token)
        last = len(emitted) >= budget
        yield StreamChunk(
            token_id=token,
            text=self.tokenizer.decode([token]),
            prompt_tokens=prompt_tokens,
            finished=last,
            finish_reason="length" if last else None,
        )
        return "length" if last else None

    def _final(self, prompt_tokens: int, finish_reason: str) -> StreamChunk:
        return StreamChunk(
            token_id=None,
            text="",
            prompt_tokens=prompt_tokens,
            finished=True,
            finish_reason=finish_reason,
        )
