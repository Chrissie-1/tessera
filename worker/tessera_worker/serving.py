"""Phase 2: the continuous batcher, as a servable engine.

`ContinuousBatcher` is a scheduler: it advances every running sequence by one
token each time someone calls `step`. Nobody was calling it. The gRPC servicer
handed each request to `engine.generate` on a thread pool, so concurrent
requests ran as concurrent single-sequence decodes and the batching win never
reached production.

This wraps the scheduler in a background loop and exposes the same interface
the servicer already uses -- `generate` and `iter_generate` -- so requests
arriving on different threads are merged into one batched forward pass without
the serving layer knowing that anything changed.

The model is touched only by the scheduler thread. Request threads submit work
and consume tokens through queues, which keeps the tensor work single-threaded
without a lock around the forward pass.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator

from .batching import ContinuousBatcher, Request
from .config import WorkerConfig
from .model import GenerationResult
from .paged_engine import PagedEngine, StreamChunk
from .sampling import SamplingParams

logger = logging.getLogger(__name__)

# How long the scheduler blocks waiting for work before looping. It only
# bounds shutdown latency; arriving requests wake it immediately.
_IDLE_TIMEOUT_SECONDS = 0.05


class _Closed:
    """Sentinel marking the end of a request's token stream."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error


class BatchedEngine:
    """Serves many concurrent requests through one continuous-batching loop."""

    def __init__(
        self,
        config: WorkerConfig,
        max_batch_size: int | None = None,
        num_blocks: int | None = None,
        block_size: int | None = None,
    ) -> None:
        max_batch_size = (
            max_batch_size if max_batch_size is not None else config.max_batch_size
        )
        self.engine = PagedEngine(config, num_blocks=num_blocks, block_size=block_size)
        self.batcher = ContinuousBatcher(self.engine, max_batch_size=max_batch_size)

        self._condition = threading.Condition()
        self._queues: dict[str, queue.SimpleQueue] = {}
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="tessera-scheduler", daemon=True
        )
        self._thread.start()

    # -- delegation ---------------------------------------------------------
    # The servicer and the HTTP layer reach through the engine for these.

    @property
    def config(self) -> WorkerConfig:
        return self.engine.config

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def model(self):
        return self.engine.model

    @property
    def eos_token_id(self) -> int | None:
        return self.engine.eos_token_id

    @property
    def max_position_embeddings(self) -> int:
        return self.engine.max_position_embeddings

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Stop the scheduler thread. Safe to call more than once."""
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> BatchedEngine:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- the scheduler loop -------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stopping and not self.batcher.has_work:
                    self._condition.wait(timeout=_IDLE_TIMEOUT_SECONDS)
                if self._stopping:
                    return

            try:
                outputs = self.batcher.step()
            except BaseException as exc:  # noqa: BLE001 - relayed to callers
                # A step advances every running sequence at once, so a failure
                # belongs to all of them. Failing them explicitly is what stops
                # their threads blocking on a queue forever.
                logger.exception("scheduler step failed")
                self._fail_all(exc)
                continue

            for output in outputs:
                sink = self._queues.get(output.request_id)
                if sink is not None:
                    sink.put(output)
                    if output.finished:
                        sink.put(_Closed())

    def _fail_all(self, exc: BaseException) -> None:
        with self._condition:
            sinks = list(self._queues.values())
            self._queues.clear()
            self.batcher.waiting.clear()
            for sequence in self.batcher.running:
                self.engine.cache.free_sequence(sequence.seq_id)
            self.batcher.running.clear()
        for sink in sinks:
            sink.put(_Closed(exc))

    # -- engine interface ---------------------------------------------------

    def iter_generate(
        self,
        prompt: str,
        max_tokens: int,
        params: SamplingParams | None = None,
    ) -> Iterator[StreamChunk]:
        """Submit one request and yield its tokens as the scheduler produces them."""
        request_id = str(uuid.uuid4())
        sink: queue.SimpleQueue = queue.SimpleQueue()

        with self._condition:
            if self._stopping:
                raise RuntimeError("engine is shutting down")
            self._queues[request_id] = sink
            self.batcher.submit(
                Request(
                    request_id=request_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    params=params or SamplingParams(),
                )
            )
            self._condition.notify()

        try:
            while True:
                item = sink.get()
                if isinstance(item, _Closed):
                    if item.error is not None:
                        raise item.error
                    return
                yield StreamChunk(
                    token_id=item.token_id,
                    text=item.text,
                    prompt_tokens=item.prompt_tokens,
                    finished=item.finished,
                    finish_reason=item.finish_reason,
                )
        finally:
            with self._condition:
                self._queues.pop(request_id, None)

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
