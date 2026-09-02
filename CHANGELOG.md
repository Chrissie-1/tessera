# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`metrics.py`** — Prometheus metrics for the telemetry the engines were
  already computing and then discarding. Counters for requests
  (`tessera_requests_total{outcome}`, where `outcome` is `success`, `error` or
  `rejected`), prompt and generated tokens, and a latency histogram
  (`tessera_request_latency_seconds`) recorded on both the gRPC and the HTTP
  paths, streaming included. Gauges for instantaneous state, read off the live
  objects at scrape time rather than mirrored from the decode loop: the
  servicer's in-flight count, the paged allocator's block totals and
  `PagedKVCache.utilisation`, the `ContinuousBatcher` queue (waiting, running,
  pending, slot limit), and `SpeculativeEngine`'s proposed/accepted counters
  with its acceptance rate. Metrics with no source in the running backend are
  absent rather than zero, and nothing is labelled by prompt, request id or
  generated text.
- **`GET /metrics` on the FastAPI wrapper**, and a standalone Prometheus HTTP
  exporter for the gRPC worker, since production runs gRPC and had no HTTP
  surface at all. The exporter is opt-in through the new
  `TESSERA_METRICS_PORT` (default `0`, off), because the gRPC server is also
  started embedded by tests and by the benchmark harness; a port it cannot
  bind is logged and skipped rather than raised.
- **`worker/tests/test_metrics.py`** — the exported numbers checked against
  the state they describe: the token counters against what the servicer
  returned, the cache gauges against the allocator's own block count
  mid-decode and after the sequence is freed, the queue gauges against the
  scheduler's own deque, the speculative counters against the engine's
  attributes. Also that a failing counter does not fail a request and a
  failing collector does not fail a scrape.
- **`prometheus-client`** is a declared dependency again, this time because
  `metrics.py` imports it.
- **`attention_hook.py`** — paged attention is now on the decode path.
  `PagedEngine`'s single-token steps register a function in transformers'
  `ALL_ATTENTION_FUNCTIONS` and select it on the model, so each layer writes
  the new token's keys and values into the block pool and attends over the
  block table through the `paged_attention` dispatcher. `PagedKVCache.gather`
  is no longer called during decode.
- **`PagedKVCache.reserve` / `write` / `layer_keys` / `layer_values` /
  `block_table_tensor`** — the cache API attention writes through, for a
  forward pass that fills the pool one layer at a time rather than all at once.
- **`PagedEngine.decode_step`** — one decode step, on either the paged or the
  gathered route, so the two cannot drift.
- **Grouped-query attention on the paged path** — `paged_attention_torch` and
  the Triton kernel now map each group of query heads onto its shared KV head,
  so a model with fewer KV heads than attention heads decodes through the block
  table instead of being deferred. Neither implementation expands the keys.
  `kv_group_size` rejects head counts that do not divide evenly rather than
  guessing a mapping.
- **`worker/tests/test_architectures.py`** — the decode stack run end to end
  against three non-GPT-2 families on tiny random hub checkpoints: Llama
  (rotary positions), Mistral (rotary plus grouped-query attention, 4 query
  heads onto 2 KV heads) and GPT-NeoX (partial rotary, parallel residual).
  Reference, paged (kernel path and gather fallback), continuous batching
  (the scheduler directly and `BatchedEngine` under concurrent threads) and
  speculative decoding are each held to reference-identical token ids. In the
  default test run, not behind `slow`.

### Fixed

- **Deferred attention shapes ran with no mask at all.** Selecting a custom
  implementation through `ALL_ATTENTION_FUNCTIONS` makes transformers skip mask
  construction entirely — it assumes a vLLM-style kernel masks internally — so
  every shape `attention_hook` defers back to `sdpa_attention_forward` received
  `attention_mask=None`. Continuous batching therefore attended to its own left
  padding, and speculative verification aligned its causal mask to the top left
  of a non-square score matrix. Both silently changed the output. The
  implementation name is now registered against `sdpa_mask` in
  `ALL_MASK_ATTENTION_FUNCTIONS`, so the fallback gets exactly the mask plain
  `sdpa` would have. This was never Llama-specific: it affects GPT-2 too, and
  went unnoticed only because `sshleifer/tiny-gpt2` has two heads of one
  dimension each, which leaves its argmax indifferent to what attention returns.
- **Grouped-query models decoded against a one-token context.** The hook
  deferred GQA to the model's own attention, but `PagedEngine.decode_step`
  hands the model no `past_key_values` when the hook is installed, so the
  deferral attended over the new token alone and wrote nothing into the pool.
  The output was wrong rather than merely slow. Fixed by covering GQA in the
  kernel; the `decode_step` fast path now has no reachable deferral.
- **Sliding-window models are declined rather than answered wrongly.** The
  block-table walk attends to every cached position, so a model that should
  forget the start of its context would remember it. `enable_paged_attention`
  now keeps such a model on the gather path — but only when the window is
  narrower than the model's own position limit, so a window that can never bind
  costs nothing.

### Notes

- Prefill and batched decode still deliberately stay on the model's own
  attention; an uninstallable hook falls back to gathering.
- The integration is verified on CPU, where the dispatcher selects the torch
  implementation. It has never been run on CUDA, so the Triton kernel's
  behaviour *on the decode path* is untested — including its grouped-query
  head mapping, which is checked against the torch implementation only by
  GPU-marked tests that skip without a device.
- The architecture coverage is on randomly-initialised checkpoints of one to
  two million parameters. It proves head-count resolution, position handling,
  mask construction and block-table indexing; it says nothing about output
  quality, and no model of a realistic size has been decoded through these
  engines.

## [0.1.0] - 2026-09-01

First tagged release. Everything below is new.

### Worker (Python)

- **`ReferenceEngine`** — dense forward pass per token with no KV cache,
  O(n²) in sequence length. Deliberately unoptimised: it is the oracle every
  other engine is tested against.
- **`PagedKVCache` / `BlockAllocator`** — block-paged key/value storage with
  per-sequence block tables, lazy block commitment, LIFO free list, and
  `truncate` for discarding rejected speculative tokens. Pool exhaustion
  raises `OutOfBlocksError` instead of evicting.
- **`PagedEngine`** — single-sequence decoding against the paged cache, with a
  shared `iter_generate` loop backing both unary and streaming generation.
- **`ContinuousBatcher`** — iteration-level scheduler. One batched forward pass
  advances every running sequence by one token; finished sequences retire
  immediately and queued requests fill the freed slots on the next step.
  Left-padding plus explicit position ids keep mixed-length caches correct.
- **`BatchedEngine`** — the scheduler wrapped in a background thread and
  exposed through the same `generate` / `iter_generate` interface, so
  concurrent requests merge into shared forward passes. Model work stays on one
  thread; request threads communicate over queues.
- **`SpeculativeEngine`** — draft-and-verify decoding. Exact under greedy
  (a proposal is kept only if it is the target's argmax) and exact under
  sampling (accept with `min(1, p/q)`, resolve rejections from the normalised
  residual `max(0, p − q)`). Falls back to paged decoding when the drafter
  cannot report its proposal distribution. Tracks an acceptance rate.
- **`attention.py`** — paged attention over a block table, in two
  implementations: a readable PyTorch version and a Triton kernel using an
  online softmax that never materialises the context contiguously. Not yet
  wired into the decode path; validated standalone.
- **Sampling** — temperature, top-p nucleus filtering, seeded generators, and
  the residual distribution used by speculative sampling. Shared by every
  engine so output differences can never be blamed on sampling.
- **gRPC server** — `Generate`, `GenerateStream` (one message per token), and
  `Health` (readiness, model, device, in-flight count). Request validation
  against a `max_tokens` cap.
- **FastAPI dev wrapper** — `GET /health` and `POST /v1/completions`, unary or
  server-sent events, for exercising the worker without the gateway.
- **Architecture-agnostic config resolution** — layer count and position limits
  are read through whichever attribute names a model family uses, so GPT-2,
  Llama, Mistral and OPT-style configs all load.
- Backend selected at startup by `TESSERA_BACKEND`; configuration resolved from
  the environment into a frozen `WorkerConfig`. Engine sizing
  (`TESSERA_MAX_BATCH_SIZE`, `TESSERA_NUM_BLOCKS`, `TESSERA_BLOCK_SIZE`,
  `TESSERA_LOOKAHEAD`) is configurable, with explicit constructor arguments
  taking precedence.

### Gateway (Rust)

- Axum HTTP server exposing `POST /v1/completions` (OpenAI-shaped JSON) and
  `GET /health` (gateway liveness plus per-worker readiness probes).
- Server-sent-event streaming that forwards worker token deltas and holds the
  worker lease for the life of the stream.
- Worker pool with lazy connections, least-in-flight routing via
  compare-and-swap, RAII leases that cannot leak capacity, and load shedding
  when every worker is saturated.
- Request validation ahead of dispatch, and gRPC status → HTTP status mapping.

### Infrastructure

- Protobuf service definition; Python stubs generated by `scripts/gen_proto.sh`,
  Rust stubs by `gateway/build.rs`.
- CPU Docker images for both services and a `docker compose` stack that serves
  with continuous batching and tolerates `--scale worker=N`.
- Makefile covering venv, install, proto, test, lint, fmt, run, and docker.
- 176 Python tests and 13 Rust tests, including equivalence tests asserting the
  paged, batched, and speculative engines reproduce the reference engine's
  tokens exactly, and that a streamed completion reproduces the unary one.
- `bench/run.py` (engine throughput comparison) and `bench/attention.py`
  (kernel vs. gather across context lengths, CUDA only).

### Known limitations

- The Triton paged-attention kernel is not wired into the serving path.
- Config resolution covers the common architectures, but only GPT-2 has been
  run end to end here; treat other families as unverified.
- `docker compose --scale worker=N` runs N workers, but Compose gives them one
  DNS name, so the gateway spreads load by resolution rather than by its own
  least-in-flight accounting.
- No metrics endpoint, authentication, TLS, or per-request timeouts.
- The default drafter is the target model itself, which proves correctness but
  cannot be faster than not speculating.

[Unreleased]: https://github.com/Chrissie-1/tessera/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Chrissie-1/tessera/releases/tag/v0.1.0
