"""Phase 2: continuous batching.

Phase 1 served requests on a thread pool, so a slot was held for a request's
entire decode. A 500-token completion blocked that slot for 500 steps even
though each step is a tiny amount of work, and short requests queued behind
long ones.

This scheduler batches at the granularity of a single decode step instead. On
every iteration each running sequence contributes one token to one batched
forward pass; sequences that finish are evicted immediately and waiting
requests are admitted into the freed slots on the very next step, rather than
waiting for the whole batch to drain.

Sequences in a batch have different lengths, so their caches are left-padded to
a common width and masked. Left rather than right padding is what keeps the
newest token at a fixed offset for every row, so one set of position ids
describes the whole batch.
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from .paged import OutOfBlocksError
from .paged_engine import PagedEngine
from .sampling import SamplingParams, make_generator, sample_token

logger = logging.getLogger(__name__)


@dataclass
class Request:
    """A unit of work handed to the scheduler."""

    request_id: str
    prompt: str
    max_tokens: int
    params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class Sequence:
    """A request that has been admitted and is decoding."""

    request: Request
    seq_id: int
    prompt_tokens: int
    tokens: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    generator: torch.Generator | None = None

    @property
    def done(self) -> bool:
        return self.finish_reason is not None

    @property
    def request_id(self) -> str:
        return self.request.request_id


@dataclass
class StepOutput:
    """What one scheduler iteration produced for one sequence."""

    request_id: str
    token_id: int | None
    text: str
    finished: bool
    finish_reason: str | None


class ContinuousBatcher:
    """Iteration-level scheduler over a shared paged cache.

    Admission is bounded by two things: a slot limit, which caps the batch
    width, and free blocks in the cache. A request that cannot be given blocks
    stays queued rather than displacing a running sequence, so admitted work
    always runs to completion.
    """

    def __init__(self, engine: PagedEngine, max_batch_size: int = 8) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Sequence] = []
        self._ids = itertools.count()

    # -- queue ---------------------------------------------------------------

    def submit(self, request: Request) -> None:
        self.waiting.append(request)

    @property
    def pending(self) -> int:
        return len(self.waiting) + len(self.running)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # -- admission -----------------------------------------------------------

    def _admit(self) -> list[Sequence]:
        """Fill free slots from the queue, prefilling each new sequence."""
        admitted: list[Sequence] = []
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting[0]
            prompt_ids = self.engine.tokenizer.encode(request.prompt)
            if not prompt_ids:
                prompt_ids = [self.engine.eos_token_id or 0]

            seq_id = next(self._ids)
            self.engine.cache.add_sequence(seq_id)
            try:
                logits = self._prefill(seq_id, prompt_ids)
            except OutOfBlocksError:
                # Out of memory, not out of turn: leave it queued and retry
                # once a running sequence frees its blocks.
                self.engine.cache.free_sequence(seq_id)
                logger.info("admission deferred, cache full")
                break

            self.waiting.popleft()
            sequence = Sequence(
                request=request,
                seq_id=seq_id,
                prompt_tokens=len(prompt_ids),
                generator=make_generator(
                    self.engine.config.device, request.params.seed
                ),
            )
            self._advance(sequence, logits)
            self.running.append(sequence)
            admitted.append(sequence)
        return admitted

    @torch.inference_mode()
    def _prefill(self, seq_id: int, prompt_ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.engine.config.device
        )
        outputs = self.engine.model(input_ids=input_ids, use_cache=True)
        self.engine.cache.append(seq_id, self.engine.new_kv(outputs.past_key_values, 0))
        return outputs.logits[0, -1, :]

    # -- decode --------------------------------------------------------------

    def _advance(self, sequence: Sequence, logits: torch.Tensor) -> int | None:
        """Sample one token and apply the stop conditions."""
        token = sample_token(logits, sequence.request.params, sequence.generator)

        if token == self.engine.eos_token_id:
            sequence.finish_reason = "stop"
            return None

        sequence.tokens.append(token)
        if len(sequence.tokens) >= sequence.request.max_tokens:
            sequence.finish_reason = "length"
        return token

    @torch.inference_mode()
    def _batched_logits(self, batch: list[Sequence], tokens: list[int]) -> torch.Tensor:
        """One forward pass advancing every sequence in `batch` by one token.

        Caches of differing length are left-padded to a common width; the
        attention mask hides the padding, and explicit position ids stop the
        padding from shifting each sequence's notion of where it is.
        """
        cache = self.engine.cache
        lengths = [cache.length(s.seq_id) for s in batch]
        width = max(lengths)
        num_layers = cache.num_layers

        per_sequence = [cache.gather(s.seq_id) for s in batch]
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(num_layers):
            keys, values = [], []
            for gathered, length in zip(per_sequence, lengths, strict=True):
                key, value = gathered[layer]
                pad = width - length
                if pad:
                    key = torch.nn.functional.pad(key, (0, 0, pad, 0))
                    value = torch.nn.functional.pad(value, (0, 0, pad, 0))
                keys.append(key)
                values.append(value)
            layers.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        device = self.engine.config.device
        mask = torch.zeros(len(batch), width + 1, dtype=torch.long, device=device)
        for row, length in enumerate(lengths):
            mask[row, width - length :] = 1

        outputs = self.engine.model(
            input_ids=torch.tensor(
                [[t] for t in tokens], dtype=torch.long, device=device
            ),
            past_key_values=DynamicCache(layers),
            attention_mask=mask,
            position_ids=torch.tensor(
                [[length] for length in lengths], dtype=torch.long, device=device
            ),
            use_cache=True,
        )

        # Write each row's new key/value back to its own block table. The
        # batched cache is scratch; the paged cache is the durable copy.
        produced = outputs.past_key_values
        for row, sequence in enumerate(batch):
            cache.append(
                sequence.seq_id,
                [
                    (
                        layer.keys[row : row + 1, :, -1:, :],
                        layer.values[row : row + 1, :, -1:, :],
                    )
                    for layer in produced.layers
                ],
            )
        return outputs.logits[:, -1, :]

    # -- the loop ------------------------------------------------------------

    def step(self) -> list[StepOutput]:
        """Run one scheduler iteration.

        Admits what it can, advances every running sequence by one token, and
        retires those that finished. Returns one entry per sequence that
        produced something this iteration.
        """
        outputs: list[StepOutput] = []

        # A sequence admitted this iteration already produced its first token
        # from the prefill logits, so it sits out the batched decode until the
        # next one. That keeps the invariant at one token per sequence per step.
        admitted = self._admit()
        just_admitted = {id(s) for s in admitted}
        for sequence in admitted:
            outputs.append(self._emit(sequence))

        batch = [s for s in self.running if not s.done and id(s) not in just_admitted]
        if batch:
            tokens = [s.tokens[-1] for s in batch]
            logits = self._batched_logits(batch, tokens)
            for row, sequence in enumerate(batch):
                self._advance(sequence, logits[row])
                outputs.append(self._emit(sequence))

        self._retire()
        return outputs

    def _emit(self, sequence: Sequence) -> StepOutput:
        token = sequence.tokens[-1] if sequence.tokens else None
        return StepOutput(
            request_id=sequence.request_id,
            token_id=token,
            text=self.engine.tokenizer.decode([token]) if token is not None else "",
            finished=sequence.done,
            finish_reason=sequence.finish_reason,
        )

    def _retire(self) -> None:
        """Free finished sequences so their blocks are reusable immediately."""
        still_running = []
        for sequence in self.running:
            if sequence.done:
                self.engine.cache.free_sequence(sequence.seq_id)
            else:
                still_running.append(sequence)
        self.running = still_running

    def run_to_completion(self) -> dict[str, list[int]]:
        """Drain every queued and running request. Returns tokens per request.

        Convenience for tests and batch jobs; a server calls `step` in its own
        loop so it can stream each token as it appears.
        """
        collected: dict[str, list[int]] = {}
        while self.has_work:
            for output in self.step():
                if output.token_id is not None:
                    collected.setdefault(output.request_id, []).append(output.token_id)
        return collected
