"""Phase 2: decoding on top of the paged KV cache.

`ReferenceEngine` re-reads the whole sequence on every step, so generating n
tokens costs O(n^2) forward passes over the prompt. `PagedEngine` keeps the
attention keys and values in `PagedKVCache` and feeds the model one new token
per step, which makes each step O(1) in the prompt length.

It deliberately subclasses the reference engine rather than reimplementing it.
Model loading, tokenisation and the sampling call are inherited unchanged, so
when `test_paged.py` asserts the two engines emit identical tokens, the only
thing that assertion can be testing is the cache.

Unary and streaming generation share one decode loop -- `iter_generate` -- so
the two cannot drift apart. `generate` is that loop, accumulated.

Decode steps read the cache through `attention_hook`, so the block table is
walked inside attention and the per-step gather disappears. Prefill is left on
the model's own attention deliberately -- it is one dense pass, not a per-token
cost -- and if the hook cannot be installed the engine falls back to gathering,
which is slower and identical.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from .attention_hook import enable_paged_attention, paged_decode
from .config import WorkerConfig, num_layers
from .model import GenerationResult, ReferenceEngine
from .paged import PagedKVCache
from .sampling import SamplingParams, make_generator, sample_token

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """One token's worth of progress.

    `text` is the delta for this token alone, not the completion so far, so a
    caller can forward it straight to the client without diffing.
    """

    token_id: int | None
    text: str
    prompt_tokens: int
    finished: bool
    finish_reason: str | None


class PagedEngine(ReferenceEngine):
    """Single-sequence decoding against a block-paged cache."""

    def __init__(
        self,
        config: WorkerConfig,
        num_blocks: int | None = None,
        block_size: int | None = None,
    ) -> None:
        super().__init__(config)
        # An explicit argument wins; otherwise take the configured value, which
        # itself defaults to the module constant.
        num_blocks = num_blocks if num_blocks is not None else config.num_blocks
        block_size = block_size if block_size is not None else config.block_size
        self.cache = PagedKVCache(
            num_layers=num_layers(self.model.config),
            num_blocks=num_blocks,
            block_size=block_size,
            device=config.device,
            dtype=config.dtype,
        )
        self._next_seq_id = 0
        # Whether single-token decode reads the pool through the paged kernel.
        # Recorded rather than re-checked per step, and consulted by the tests
        # that assert the kernel is genuinely on the decode path.
        self.paged_attention_enabled = enable_paged_attention(self.model)

    def _new_sequence_id(self) -> int:
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        return seq_id

    @staticmethod
    def new_kv(
        cache: DynamicCache, since: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Slice out only the positions this forward pass added.

        The model returns the full cache every step; writing all of it back
        would duplicate everything already paged.
        """
        return [
            (layer.keys[:, :, since:, :], layer.values[:, :, since:, :])
            for layer in cache.layers
        ]

    def decode_step(self, seq_id: int, token_id: int) -> torch.Tensor:
        """Advance `seq_id` by one token and return the next position's logits.

        Two routes to the same answer. The paged route claims the position
        first, then lets attention write into it and read the block table
        directly, so no copy of the context is made. The gathered route is the
        Phase 2 behaviour, kept for whenever the hook is not installed.
        """
        input_ids = torch.tensor(
            [[token_id]], dtype=torch.long, device=self.config.device
        )

        if self.paged_attention_enabled:
            position = self.cache.reserve(seq_id, 1)
            with paged_decode(self.cache, seq_id, position):
                # No `past_key_values`: the past lives in the pool, and the
                # model would otherwise concatenate a second copy of it. That
                # also means the position has to be stated, since there is no
                # cache length for the model to infer it from. `position_ids`
                # is enough for the learned embeddings GPT-2 looks them up in
                # and for the rotary embeddings Llama, Mistral and GPT-NeoX
                # compute from them, which `test_architectures.py` covers.
                # An architecture that positioned itself from `cache_position`
                # alone would need that passed too.
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=False,
                    position_ids=torch.tensor(
                        [[position]], dtype=torch.long, device=self.config.device
                    ),
                )
            return outputs.logits[0, -1, :]

        cached_len = self.cache.length(seq_id)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.cache.to_cache(seq_id),
            use_cache=True,
        )
        self.cache.append(seq_id, self.new_kv(outputs.past_key_values, cached_len))
        return outputs.logits[0, -1, :]

    @torch.inference_mode()
    def iter_generate(
        self,
        prompt: str,
        max_tokens: int,
        params: SamplingParams | None = None,
    ) -> Iterator[StreamChunk]:
        """Decode against the paged cache, yielding one chunk per token.

        Mirrors `ReferenceEngine.generate` exactly -- same budget, same stop
        condition, same sampling call -- so any output difference is a cache
        bug rather than a policy difference. The final chunk always carries
        `finished`, even when the last token and the stop coincide.
        """
        params = params or SamplingParams()

        prompt_ids: list[int] = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.eos_token_id or 0]
        prompt_tokens = len(prompt_ids)

        generator = make_generator(self.config.device, params.seed)
        finish_reason = "length"
        emitted = 0

        seq_id = self._new_sequence_id()
        self.cache.add_sequence(seq_id)
        try:
            # Prefill: one dense pass over the prompt, then page the result.
            input_ids = torch.tensor(
                [prompt_ids], dtype=torch.long, device=self.config.device
            )
            outputs = self.model(input_ids=input_ids, use_cache=True)
            self.cache.append(seq_id, self.new_kv(outputs.past_key_values, 0))
            logits = outputs.logits[0, -1, :]

            budget = min(max_tokens, self.max_position_embeddings - prompt_tokens)
            for _ in range(max(0, budget)):
                next_token = sample_token(logits, params, generator)
                if next_token == self.eos_token_id:
                    finish_reason = "stop"
                    break

                emitted += 1
                last = emitted >= max(0, budget)
                yield StreamChunk(
                    token_id=next_token,
                    text=self.tokenizer.decode([next_token]),
                    prompt_tokens=prompt_tokens,
                    finished=last,
                    finish_reason="length" if last else None,
                )
                if last:
                    return

                logits = self.decode_step(seq_id, next_token)

            # Reached only by an EOS stop or a zero-token budget, both of
            # which still owe the caller a terminal chunk.
            yield StreamChunk(
                token_id=None,
                text="",
                prompt_tokens=prompt_tokens,
                finished=True,
                finish_reason=finish_reason,
            )
        finally:
            # Blocks go back even if the model raises, or the pool leaks and
            # the worker degrades until restart.
            self.cache.free_sequence(seq_id)

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        params: SamplingParams | None = None,
    ) -> GenerationResult:
        """Accumulate `iter_generate` into a single result."""
        start = time.perf_counter()

        generated: list[int] = []
        prompt_tokens = 0
        finish_reason = "length"

        for chunk in self.iter_generate(prompt, max_tokens, params):
            prompt_tokens = chunk.prompt_tokens
            if chunk.token_id is not None:
                generated.append(chunk.token_id)
            if chunk.finished:
                finish_reason = chunk.finish_reason or finish_reason

        return GenerationResult(
            text=self.tokenizer.decode(generated),
            token_ids=generated,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(generated),
            finish_reason=finish_reason,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
