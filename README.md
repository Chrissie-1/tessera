# Tessera

An LLM inference stack built to make each optimisation prove itself: a Rust HTTP
gateway in front of a Python worker that decodes with a paged KV cache,
continuous batching, and speculative decoding.

The organising idea is a **reference engine that is deliberately slow**.
`ReferenceEngine` re-runs a full dense forward pass for every token, keeps no
cache, and is never optimised. Every faster engine in the repo is required by
the test suite to emit *the same token ids* as that reference. A speedup that
changes the output is a bug, not a speedup.

## Status

This is a working, well-tested implementation of the decoding techniques below.
It is a learning and benchmarking project, not a production serving system —
see [Limitations](#limitations) before deploying it anywhere that matters.

## Architecture

```
client
  │  HTTP  POST /v1/completions        (JSON, or SSE when "stream": true)
  ▼
┌─────────────────────────┐
│  gateway (Rust, :8080)  │  validation · least-in-flight routing · load shedding
└─────────────────────────┘
  │  gRPC  Generate / GenerateStream / Health
  ▼
┌─────────────────────────┐
│  worker (Python, :50051)│  tokenise · decode · KV cache
└─────────────────────────┘
  │
  ▼
transformers / PyTorch
```

The gateway holds no model state. It validates requests, picks the worker with
the fewest in-flight requests, sheds load when every worker is saturated, and
forwards token streams as server-sent events.

### Decoding backends

The worker selects one backend at startup via `TESSERA_BACKEND`:

| Backend | Class | What it does |
|---|---|---|
| `reference` *(default)* | `ReferenceEngine` | Dense forward pass per token, no cache. O(n²) in sequence length. Defines correctness. |
| `paged` | `PagedEngine` | Block-paged KV cache; one new token per forward pass. |
| `batched` | `BatchedEngine` | Runs `paged` under a continuous-batching scheduler on a background thread, so concurrent requests share forward passes. |
| `speculative` | `SpeculativeEngine` | Draft-and-verify on top of `paged`. |

`reference` is the default because it is the one engine that cannot be wrong.
For anything throughput-sensitive, set `TESSERA_BACKEND=batched`.

Only `paged`, `batched`, and `speculative` implement `iter_generate`, so
`GenerateStream` returns `UNIMPLEMENTED` on the `reference` backend.

### Paged KV cache

A contiguous per-sequence cache forces every sequence to reserve room for the
longest completion it *might* produce. `PagedKVCache` instead hands out
fixed-size blocks from a pool and gives each sequence a block table, so memory
is committed only as it is used and freeing a sequence returns whole blocks
with no compaction. Exhaustion raises `OutOfBlocksError` rather than silently
evicting, which lets the scheduler above decide whether to queue or shed.

### Continuous batching

`ContinuousBatcher` schedules at the granularity of one decode step. Each
iteration advances every running sequence by one token in a single batched
forward pass; finished sequences are retired immediately and queued requests
are admitted into the freed slots on the next step. Caches of differing length
are left-padded to a common width and masked, which keeps the newest token at a
fixed offset for every row.

### Speculative decoding

A drafter proposes *k* tokens, the target model verifies all *k* in one pass,
and accepted proposals are kept. Rejected proposals have their keys and values
truncated out of the cache before the next round.

This is exact, not approximate, in both modes:

- **Greedy** — a proposal is kept only when it equals the target's own argmax.
- **Sampling** — a proposal drawn from the drafter's `q` is accepted with
  probability `min(1, p/q)`, and a rejection is resolved by drawing from the
  normalised residual `max(0, p − q)`. Those two steps compose to exactly `p`.

Because sampling needs the proposal distribution, a drafter that cannot report
one falls back to ordinary paged decoding rather than sampling from the wrong
distribution. The tests enforce exactness by running a deliberately bad drafter
and still demanding reference-identical output.

### Paged attention kernel

`attention.py` contains two implementations of attention over a block table: a
readable PyTorch one that gathers, and a Triton kernel that walks the block
table inside the kernel with an online softmax, so a sequence's keys and values
are never materialised contiguously.

`attention_hook.py` wires it into `PagedEngine`'s decode path. transformers 5.x
resolves attention through `ALL_ATTENTION_FUNCTIONS`, so the hook registers a
function there and selects it on the model: each layer writes the new token's
keys and values straight into the block pool and attends over the block table,
and `PagedKVCache.gather` is never called during decode. The dispatcher is what
is wired in, so CUDA gets the Triton kernel and everything else gets the torch
implementation.

**Grouped-query attention is covered.** The pool stores whatever KV head count
the model produces, and the kernel folds each group of query heads onto its
shared KV head — in the torch implementation by reshaping the query, in the
Triton kernel by dividing the program's head id down. Neither expands the keys,
which would be the copy paging exists to avoid.

Two cases deliberately stay on the model's own attention: **prefill** (many
query positions, one dense pass either way, not the per-token cost the kernel
removes) and **batched decode** under `ContinuousBatcher` (one padded cache for
the whole batch). The hook also declines to install on a model whose declared
**sliding window** is narrower than its own position limit, because the
block-table walk attends to every cached position and would remember a context
the model is supposed to have forgotten. If the hook cannot be installed — a
transformers without the attention interface, or a model not on `sdpa` — the
engine falls back to gathering, which is slower and identical.

Selecting a custom attention implementation has one consequence worth naming:
transformers builds **no attention mask at all** for an implementation it does
not recognise, on the assumption that a vLLM-style kernel masks internally. The
single-token walk needs no mask, but every deferred shape does, so the hook
registers its name against `sdpa_mask` in `ALL_MASK_ATTENTION_FUNCTIONS` as
well. Without that, batched decode attends to its left padding and speculative
verification mis-aligns its causal mask.

**What is verified, and where.** The integration is tested on CPU against the
reference engine, and the tests fail if the engine silently reverts to
gathering. What is *not* verified here: the Triton kernel itself, which needs a
GPU. The GPU tests that hold it to the torch implementation skip without CUDA,
and this integration has never been executed on a CUDA device — on such a
machine the same hook would dispatch to `paged_attention_triton` instead, and
that combination is untested. See [Limitations](#limitations).

## Quick start

### Docker

```bash
docker compose up --build

curl -X POST http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

The worker downloads `gpt2` on first start; the HF cache is kept in a named
volume so a rebuild does not refetch it. The stack serves with
`TESSERA_BACKEND=batched`, so concurrent requests share forward passes.

### From source

Requirements: Python 3.10+, Rust 1.85 or newer (the lockfile pulls in an
edition-2024 crate; the gateway image pins 1.98). `protoc` is needed to
compile the proto. CUDA and Triton are optional and only used by the
GPU-marked tests and `bench/attention.py`.

```bash
make install      # venv + deps + gRPC stubs
make test         # Python and Rust suites
make run-worker   # gRPC worker on :50051
make run-gateway  # gateway on :8080 (separate terminal)
```

`make help` lists every target.

The Makefile targets assume a POSIX shell and are written for Linux/WSL. The
Python test suite itself is CPU-only and platform-independent — it can be run
directly with `cd worker && python -m pytest`.

## HTTP API

### `POST /v1/completions`

Request:

```json
{
  "prompt": "The capital of France is",
  "max_tokens": 16,
  "temperature": 0.0,
  "top_p": 1.0,
  "seed": 0,
  "stream": false
}
```

Only `prompt` is required. `temperature` of 0 (the default) means greedy
decoding. `seed` of 0 means unseeded.

Response:

```json
{
  "id": "9f8c...",
  "object": "text_completion",
  "choices": [
    { "index": 0, "text": " Paris.", "finish_reason": "stop" }
  ],
  "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
}
```

`finish_reason` is `"length"` (hit `max_tokens`) or `"stop"` (hit EOS).
`text` is the continuation only — it does not include the prompt.

With `"stream": true` the response is `text/event-stream`, one event per token,
each carrying a `text_completion.chunk` object whose `choices[0].text` is that
token's delta. The stream ends with a `[DONE]` event.

Errors are returned as `{"error": {"message": "..."}}`. Requests are rejected
with 400 for a non-positive `max_tokens` or one above the cap, and 503 when
every worker is at capacity.

### `GET /health`

Returns gateway liveness plus a per-worker readiness probe:

```json
{
  "status": "ok",
  "workers": [
    {
      "endpoint": "http://worker:50051",
      "in_flight": 0,
      "ready": true,
      "model": "gpt2",
      "device": "cpu",
      "worker_in_flight": 0
    }
  ]
}
```

### Worker gRPC service

`proto/inference.proto` defines `Generate` (unary), `GenerateStream` (one
message per token), and `Health` (readiness plus in-flight count, which the
gateway uses for routing).

### Worker HTTP wrapper

`tessera_worker.api` is a small FastAPI app for poking at the worker without
the gateway (`make run-api`, port 8000). It exposes `GET /health` and
`POST /v1/completions`, including `"stream": true`, which emits the same SSE
shape the gateway does. It is a debugging surface, not the production path.

## Configuration

### Gateway

| Variable | Default | Description |
|---|---|---|
| `TESSERA_GATEWAY_PORT` | `8080` | HTTP listen port |
| `TESSERA_WORKER_ENDPOINTS` | `http://127.0.0.1:50051` | Comma-separated worker endpoints |
| `TESSERA_MAX_IN_FLIGHT` | `8` | Concurrent requests allowed **per worker** before shedding |
| `TESSERA_MAX_TOKENS_CAP` | `512` | Largest accepted `max_tokens` |
| `TESSERA_LOG` | `info` | `tracing` env-filter directive |

### Worker

| Variable | Default | Description |
|---|---|---|
| `TESSERA_MODEL` | `gpt2` | HuggingFace model id |
| `TESSERA_BACKEND` | `reference` | `reference`, `paged`, `batched`, or `speculative` |
| `TESSERA_DEVICE` | `auto` | `auto` picks `cuda` when available, else `cpu` |
| `TESSERA_DTYPE` | `auto` | `auto` is `float32`; set e.g. `float16` when benchmarking |
| `TESSERA_GRPC_PORT` | `50051` | gRPC listen port |
| `TESSERA_HTTP_PORT` | `8000` | Port for the FastAPI dev wrapper |
| `TESSERA_MAX_TOKENS_CAP` | `512` | Largest accepted `max_tokens` |
| `TESSERA_THREADS` | `8` | gRPC server thread-pool size |
| `TESSERA_LOG_LEVEL` | `INFO` | Python logging level |
| `TESSERA_MAX_BATCH_SIZE` | `8` | Sequences the `batched` scheduler runs concurrently |
| `TESSERA_NUM_BLOCKS` | `512` | Blocks in the paged KV pool |
| `TESSERA_BLOCK_SIZE` | `16` | Tokens per block |
| `TESSERA_LOOKAHEAD` | `4` | Tokens the `speculative` drafter proposes per round |

`float32` is the default dtype on purpose: the paged-vs-dense equivalence tests
compare logits, and fp16 accumulation differences would put "identical" out of
reach.

Every engine still accepts these as constructor arguments, and an explicit
argument wins over the configured value.

## Supported models

Four architecture families are decoded end to end by the default test run, on
CPU, through every engine — reference, paged (kernel path *and* gather
fallback), continuous batching (both the scheduler directly and `BatchedEngine`
under concurrent threads), and speculative — with every engine asserted to emit
the reference engine's token ids:

| Family | Checkpoint | What it adds |
| --- | --- | --- |
| GPT-2 | `sshleifer/tiny-gpt2` | learned position embeddings |
| Llama | `hf-internal-testing/tiny-random-LlamaForCausalLM` | rotary positions |
| Mistral | `hf-internal-testing/tiny-random-MistralForCausalLM` | rotary positions, grouped-query attention (4 query heads → 2 KV heads) |
| GPT-NeoX | `hf-internal-testing/tiny-random-GPTNeoXForCausalLM` | partial rotary, parallel attention/MLP residual |

`gpt2` (the serving default) is exercised by hand rather than by the suite.

These are the hub's own randomly-initialised test checkpoints, one to two
million parameters each. They prove the *plumbing* — head-count resolution,
position handling, mask construction, block-table indexing — on architectures
that differ where it matters. They say nothing about output quality, and
nothing about scale: no model above a few million parameters has been run
here, and none on a GPU.

Layer count and position limits are resolved through whichever attribute names
an architecture uses, so OPT-style configs load too; that resolution is
unit-tested in `test_config.py` across `GPT2Config`, `LlamaConfig` and
`MistralConfig`. Decoding is architecture-agnostic beyond this point — the
cache stores whatever key/value shapes the model produces — but treat a family
outside the table above as unverified until you have run it.

A model that declares a sliding window narrower than its own position limit
(Mistral-7B, for instance) loads and decodes correctly, but keeps the gather
path: the paged kernel attends to the whole block table and has no way to
express the window. See [Limitations](#limitations).

## Development

```bash
make test      # test-py + test-rs
make test-py   # pytest, tiny model, CPU only
make test-rs   # cargo test
make lint      # ruff + black --check + cargo fmt --check + clippy -D warnings
make fmt       # ruff --fix + black + cargo fmt
make proto     # regenerate the Python gRPC stubs
```

`make proto` regenerates the **Python** stubs only. The Rust stubs are compiled
by `gateway/build.rs` on every `cargo build`, and the generated Python stubs are
gitignored — run `make proto` (or `make install`) after cloning.

Two pytest markers are registered: `slow` (loads a real model) and `gpu`
(requires CUDA). Neither is skipped by default configuration, but the `gpu`
tests skip themselves when CUDA or Triton is absent.

### Layout

```
tessera/
├── gateway/src/
│   ├── main.rs          # config, router, startup
│   ├── pool.rs          # worker pool, least-in-flight leases, shedding
│   └── routes.rs        # /v1/completions (JSON + SSE), /health
├── worker/tessera_worker/
│   ├── config.py        # env-resolved WorkerConfig
│   ├── model.py         # ReferenceEngine — the correctness oracle
│   ├── paged.py         # BlockAllocator, PagedKVCache
│   ├── paged_engine.py  # PagedEngine, StreamChunk
│   ├── batching.py      # ContinuousBatcher scheduler
│   ├── serving.py       # BatchedEngine — scheduler as a servable engine
│   ├── speculative.py   # SpeculativeEngine, ModelDrafter
│   ├── attention.py     # paged attention: torch reference + Triton kernel
│   ├── attention_hook.py # the kernel, registered as the model's attention
│   ├── sampling.py      # temperature, top-p, residual distribution
│   ├── engine.py        # backend registry and lifecycle
│   ├── server.py        # gRPC servicer
│   └── api.py           # FastAPI dev wrapper
├── proto/inference.proto
└── bench/
    ├── run.py           # engine-vs-engine throughput
    └── attention.py     # kernel vs gather across context lengths
```

## Benchmarking

```bash
python bench/run.py --model sshleifer/tiny-gpt2 --max-tokens 32
python bench/attention.py --context 128 512 2048   # requires CUDA
```

`bench/run.py` times `reference`, `paged`, `paged_batched`, and `speculative`
on the same prompts and reports tokens/sec, speedup over the reference, and the
speculative acceptance rate.

**This repo publishes no benchmark numbers.** Results depend entirely on the
machine, model, and batch composition, so `bench/results/` is gitignored. Run
the harness on your own hardware; the tiny test model in particular is far too
small for the results to say anything about real serving.

## Limitations

- Paged attention is wired into single-sequence decode only. Prefill and
  batched decode stay on the model's own attention. Grouped-query attention is
  covered as of the current `Unreleased` changes; sliding-window models are
  not, and the hook declines to install on one it could outgrow.
- The architecture coverage is on tiny random checkpoints, on CPU. Nothing
  larger than a few million parameters has been decoded through these engines,
  so per-architecture *performance* is entirely unmeasured.
- The Triton kernel has never been run as part of that integration: this repo's
  tests run on CPU, where the dispatcher selects the torch implementation. On a
  CUDA machine the hook dispatches to `paged_attention_triton`, which is tested
  only against the torch version in isolation, never on the decode path.
- `docker compose --scale worker=N` starts N workers, but Compose gives them a
  single DNS name, so the gateway spreads load by name resolution rather than by
  its own least-in-flight accounting. For true per-worker routing, list the
  endpoints explicitly in `TESSERA_WORKER_ENDPOINTS`.
- No metrics endpoint, no authentication, no TLS, no request timeouts beyond the
  gRPC client's 300s ceiling.
- Single process per worker; no multi-GPU, quantisation, prefix caching, or
  LoRA.
- The drafter in `SpeculativeEngine` defaults to the *target model itself*,
  which demonstrates correctness but cannot be faster than not speculating.
  A real deployment needs a genuinely smaller draft model.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `make fmt && make lint &&
make test` before opening a pull request, and any change touching a decode path
must keep the reference-equivalence tests green.

## Acknowledgements

The techniques here are implementations of ideas from published work:

- [vLLM](https://github.com/vllm-project/vllm) — paged attention and the block-table design
- [Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192) — the acceptance rule and residual sampling
- [Orca (Yu et al., OSDI '22)](https://www.usenix.org/conference/osdi22/presentation/yu) — iteration-level scheduling

## License

MIT — see [LICENSE](LICENSE).
