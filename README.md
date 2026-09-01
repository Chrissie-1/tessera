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

**The kernel is not yet wired into the serving path.** The decode engines still
call `PagedKVCache.gather`. The kernel is covered by its own equivalence tests
against the PyTorch version and measurable via `bench/attention.py`, but
replacing the model's attention with it is not done. See
[Limitations](#limitations).

## Quick start

### Docker

```bash
docker compose up --build

curl -X POST http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

The worker downloads `gpt2` on first start; the HF cache is kept in a named
volume so a rebuild does not refetch it.

### From source

Requirements: Python 3.10+, Rust (the gateway image builds on 1.83; `protoc` is
needed to compile the proto). CUDA and Triton are optional and only used by the
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
`POST /v1/completions` — unary only, no streaming. It is a debugging surface,
not the production path.

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

`float32` is the default dtype on purpose: the paged-vs-dense equivalence tests
compare logits, and fp16 accumulation differences would put "identical" out of
reach.

Batch size, block count, and block size are constructor arguments
(`max_batch_size=8`, `num_blocks=512`, `block_size=16`) and are not currently
read from the environment.

## Supported models

Tested against `sshleifer/tiny-gpt2` (the test suite) and `gpt2` (the default).

**GPT-2 family only, in practice.** `PagedEngine` reads `model.config.n_layer`
and `max_position_embeddings` reads `n_positions`, both of which are GPT-2
naming. Llama-style configs (`num_hidden_layers`, `max_position_embeddings`)
will raise `AttributeError` on load. Supporting them is a config-mapping change,
not an architectural one, but it has not been done.

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

- The Triton paged-attention kernel is validated but **not wired into decoding**.
- GPT-2-family configs only (see [Supported models](#supported-models)).
- `docker compose --scale worker=N` will not work as-is: the worker publishes a
  fixed host port, and the gateway's endpoint list is static.
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
