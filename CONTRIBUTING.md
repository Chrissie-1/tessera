# Contributing to Tessera

Thanks for your interest. This is a small project; the bar is "does it work,
is it tested, does it read like the code around it."

## The one rule that matters

**`ReferenceEngine` defines correctness.** It is the slow, cacheless, dense
implementation in `worker/tessera_worker/model.py`, and it is never optimised.
Every other engine is tested to produce byte-identical token ids to it.

If you touch a decode path — the paged cache, the batcher, speculative
decoding, sampling — the equivalence tests must stay green. An optimisation
that changes the output is a bug, not an optimisation. If you believe the
reference itself is wrong, fix the reference and say so explicitly in the PR;
do not "fix" it by loosening an equivalence assertion.

## Setup

Requirements: Python 3.10+ and Rust (with `protoc` available — `tonic-build`
compiles the proto during `cargo build`). CUDA and Triton are optional and only
needed for the GPU-marked tests and `bench/attention.py`.

```bash
git clone https://github.com/Chrissie-1/tessera.git
cd tessera
make install    # venv, dependencies, gRPC stubs
make test       # should be green before you change anything
```

The generated gRPC stubs under `worker/tessera_worker/generated/` are
gitignored — `make install` (or `make proto`) creates them. Nothing works until
they exist.

The Makefile assumes a POSIX shell. On Windows the Python suite still runs
directly:

```bash
cd worker && python -m pytest
```

## Before opening a pull request

```bash
make fmt     # ruff --fix, black, cargo fmt
make lint    # ruff, black --check, cargo fmt --check, clippy -D warnings
make test    # pytest + cargo test
```

CI runs exactly these on Python 3.10–3.13 plus a Docker end-to-end check, so a
clean local run should mean a clean CI run.

- New behaviour needs a test. Bug fixes need a test that fails without the fix.
- Anything touching a decode path needs an equivalence test against the
  reference engine.
- Update `README.md` if you change configuration, the API surface, or a
  documented limitation.
- Add an entry under `## [Unreleased]` in `CHANGELOG.md`.

## Style

**Python** — `black` and `ruff` (88 columns) enforce the mechanics. Beyond
that: type hints on public functions, and Google-style docstrings that explain
*why* rather than restating the signature. The existing modules lead with a
docstring explaining what problem the module solves and what it deliberately
does not do; match that.

**Rust** — `rustfmt` and `clippy` with warnings denied. Doc comments on public
items.

Comments earn their place by explaining a decision that isn't obvious from the
code. `# increment the counter` does not.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `test:`, `chore:`, with an
optional scope.

```
feat(batching): admit queued requests on the step a sequence retires
fix(paged): release blocks when a decode raises mid-stream
```

## Testing

```bash
cd worker && python -m pytest                      # everything
cd worker && python -m pytest tests/test_paged.py  # one file
cd worker && python -m pytest -m gpu               # CUDA only; skips without it
cd worker && python -m pytest -m "not slow"        # skip real-model tests

cargo test --all
cargo test pool::                                  # one module
```

Tests default to `sshleifer/tiny-gpt2` so the suite needs no GPU and no large
download. Override with `TEST_MODEL` if you need a real model.

Two markers are registered: `slow` (loads a non-tiny model) and `gpu`
(requires CUDA). The GPU tests skip themselves when CUDA or Triton is missing.

### End to end

```bash
docker compose up --build
curl -X POST http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 8}'
```

## Benchmarking

```bash
python bench/run.py --model sshleifer/tiny-gpt2 --max-tokens 32
python bench/attention.py --context 128 512 2048   # requires CUDA
```

Results go to the gitignored `bench/results/`. **Do not commit benchmark
numbers or quote them in documentation** — they describe the machine that
produced them, not the code. If a PR claims a speedup, say what hardware and
model produced it.

## Common tasks

**Changing the gRPC contract** — edit `proto/inference.proto`, run
`make proto` to regenerate the Python stubs (Rust stubs rebuild automatically),
then update the servicer in `worker/tessera_worker/server.py` and the gateway
in `gateway/src/routes.rs`.

**Adding a decoding backend** — subclass `PagedEngine`, register it in
`BACKENDS` in `worker/tessera_worker/engine.py`, and add an equivalence test.
Implement `iter_generate` or `GenerateStream` will return `UNIMPLEMENTED`.

**Adding a dependency** — Python goes in `worker/pyproject.toml` (then
`pip install -e "worker[dev]"`); Rust goes in the workspace `Cargo.toml` and is
referenced from `gateway/Cargo.toml` with `.workspace = true`. Don't declare a
dependency you don't import.

## Good first issues

- Map non-GPT-2 model configs (`num_hidden_layers` / `max_position_embeddings`)
  so Llama- and Mistral-style models load.
- Wire the Triton paged-attention kernel into the decode path — it is written
  and tested but unused.
- Make block count, block size, and batch size configurable from the
  environment rather than constructor defaults.
- Add streaming to the FastAPI dev wrapper; the engines already support it.

## Releasing

See [RELEASING.md](RELEASING.md).

## Getting help

Open an [issue](https://github.com/Chrissie-1/tessera/issues). For security
reports, see [SECURITY.md](SECURITY.md).

## License

Contributions are licensed under the MIT License.
