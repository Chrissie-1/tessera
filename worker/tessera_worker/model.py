"""Reference inference engine: dense forward, no KV cache.

This is intentionally the slowest correct implementation in the repo. Every
optimisation added later (paged KV cache in Phase 2, speculative decoding in
Phase 3, Triton kernels in Phase 4) is validated by asserting it produces the
same tokens as this file. Do not optimise it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import WorkerConfig, max_positions
from .sampling import SamplingParams, make_generator, sample_token

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "length"
    latency_ms: float = 0.0


class ReferenceEngine:
    """Loads a causal LM and decodes one request at a time.

    Each decode step re-runs the full forward pass over the entire sequence
    with `use_cache=False`, making the cost O(n^2) in sequence length. That is
    the point: there is no cache state to get wrong, so its output defines
    correctness for the whole project.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        logger.info(
            "loading model=%s device=%s dtype=%s",
            config.model_name,
            config.device,
            config.dtype,
        )
        start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
        )
        self.model.to(config.device)
        self.model.eval()

        self.load_seconds = time.perf_counter() - start
        logger.info("model ready in %.2fs", self.load_seconds)

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def max_position_embeddings(self) -> int:
        return max_positions(self.model.config)

    @torch.inference_mode()
    def forward_logits(self, token_ids: list[int]) -> torch.Tensor:
        """Run a dense forward pass and return logits for the final position.

        Args:
            token_ids: the full sequence decoded so far.

        Returns:
            (vocab,) logits predicting the token after `token_ids`.
        """
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.config.device
        )
        outputs = self.model(input_ids=input_ids, use_cache=False)
        return outputs.logits[0, -1, :]

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        params: SamplingParams | None = None,
    ) -> GenerationResult:
        """Decode a completion for `prompt`, one dense forward per token.

        Args:
            prompt: raw text; encoded without special tokens.
            max_tokens: hard cap on generated tokens.
            params: sampling configuration; defaults to greedy.

        Returns:
            The completion text plus token accounting and a finish reason.
        """
        params = params or SamplingParams()
        start = time.perf_counter()

        prompt_ids: list[int] = self.tokenizer.encode(prompt)
        if not prompt_ids:
            # An empty prompt has no position to predict from; seed with EOS
            # so the model has a valid single-token context.
            prompt_ids = [self.eos_token_id or 0]

        generator = make_generator(self.config.device, params.seed)
        token_ids = list(prompt_ids)
        generated: list[int] = []
        finish_reason = "length"

        budget = min(max_tokens, self.max_position_embeddings - len(prompt_ids))
        for _ in range(max(0, budget)):
            logits = self.forward_logits(token_ids)
            next_token = sample_token(logits, params, generator)

            if next_token == self.eos_token_id:
                finish_reason = "stop"
                break

            token_ids.append(next_token)
            generated.append(next_token)

        return GenerationResult(
            text=self.tokenizer.decode(generated),
            token_ids=generated,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated),
            finish_reason=finish_reason,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
