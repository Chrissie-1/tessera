# Tessera developer commands.
#
# Run these from a Linux shell (WSL is fine: cd /mnt/c/projects/tessera).
# Triton and CUDA have no Windows support, so the worker toolchain is Linux-only.

VENV ?= $(HOME)/.venvs/tessera
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
CARGO_TARGET_DIR ?= $(HOME)/.cache/tessera-target
export CARGO_TARGET_DIR

.DEFAULT_GOAL := help
.PHONY: help venv install proto test test-py test-rs lint fmt run-worker run-api run-gateway docker-build docker-up docker-down clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the Python virtualenv
	python3 -m venv $(VENV)

install: venv proto ## Install Python deps and generate gRPC stubs
	$(PIP) install --upgrade pip
	$(PIP) install -e "worker[dev]"

proto: ## Regenerate Python gRPC stubs from proto/inference.proto
	PATH="$(VENV)/bin:$$PATH" ./scripts/gen_proto.sh

test: test-py test-rs ## Run the full test suite

test-py: ## Run worker tests (tiny model, CPU only)
	cd worker && $(PY) -m pytest

test-rs: ## Run gateway tests
	cargo test

lint: ## Lint Python and Rust
	$(VENV)/bin/ruff check worker
	$(VENV)/bin/black --check worker
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings

fmt: ## Auto-format Python and Rust
	$(VENV)/bin/ruff check --fix worker
	$(VENV)/bin/black worker
	cargo fmt --all

run-worker: ## Start the gRPC worker on :50051
	cd worker && $(PY) -m tessera_worker.server

run-api: ## Start the FastAPI dev wrapper on :8000 (bypasses the gateway)
	cd worker && $(VENV)/bin/uvicorn tessera_worker.api:app --port 8000

run-gateway: ## Start the Rust gateway on :8080
	cargo run --package tessera-gateway

docker-build: ## Build both container images
	docker compose build

docker-up: ## Bring up the full stack
	docker compose up

docker-down: ## Tear down the stack and its volumes
	docker compose down -v

clean: ## Remove build artefacts and caches
	rm -rf $(CARGO_TARGET_DIR) worker/.pytest_cache worker/*.egg-info
	find worker -name __pycache__ -type d -exec rm -rf {} +
