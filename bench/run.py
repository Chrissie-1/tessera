"""Benchmark the decode paths against each other.

The equivalence tests prove the optimised engines produce the reference
engine's tokens. This measures whether they are actually worth running, which
is a separate question and the only reason the later phases exist.

Numbers here are latency per request on one process with no concurrency,
except `paged_batched`, which reports the same work run through the continuous
batching scheduler so throughput can be compared against serial decoding.

    python bench/run.py --model sshleifer/tiny-gpt2 --max-tokens 32

Results land in bench/results/ as JSON, which is gitignored: they describe the
machine that produced them and are not a property of the code.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "worker"))

import torch
from tessera_worker.batching import ContinuousBatcher, Request
from tessera_worker.config import WorkerConfig
from tessera_worker.model import ReferenceEngine
from tessera_worker.paged_engine import PagedEngine
from tessera_worker.speculative import SpeculativeEngine

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "In a shocking finding, scientists discovered",
    "The quick brown fox jumps over",
]


@dataclass
class Result:
    engine: str
    requests: int
    tokens: int
    wall_seconds: float
    tokens_per_second: float
    median_latency_ms: float
    speedup_vs_reference: float | None = None
    acceptance_rate: float | None = None


def time_serial(name: str, engine, prompts: list[str], max_tokens: int) -> Result:
    latencies: list[float] = []
    tokens = 0
    start = time.perf_counter()
    for prompt in prompts:
        began = time.perf_counter()
        result = engine.generate(prompt, max_tokens=max_tokens)
        latencies.append((time.perf_counter() - began) * 1000.0)
        tokens += result.completion_tokens
    wall = time.perf_counter() - start

    return Result(
        engine=name,
        requests=len(prompts),
        tokens=tokens,
        wall_seconds=wall,
        tokens_per_second=tokens / wall if wall else 0.0,
        median_latency_ms=statistics.median(latencies),
        acceptance_rate=getattr(engine, "acceptance_rate", None),
    )


def time_batched(engine: PagedEngine, prompts: list[str], max_tokens: int) -> Result:
    batcher = ContinuousBatcher(engine, max_batch_size=len(prompts))
    for i, prompt in enumerate(prompts):
        batcher.submit(Request(str(i), prompt, max_tokens=max_tokens))

    start = time.perf_counter()
    collected = batcher.run_to_completion()
    wall = time.perf_counter() - start

    tokens = sum(len(t) for t in collected.values())
    return Result(
        engine="paged_batched",
        requests=len(prompts),
        tokens=tokens,
        wall_seconds=wall,
        tokens_per_second=tokens / wall if wall else 0.0,
        # Every request shares one wall clock, so a per-request median would
        # describe the batch, not a request.
        median_latency_ms=wall * 1000.0 / len(prompts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "bench" / "results")
    args = parser.parse_args()

    config = WorkerConfig(
        model_name=args.model,
        device=args.device,
        dtype=torch.float32,
        grpc_port=0,
        http_port=0,
        max_tokens_cap=max(args.max_tokens, 1),
    )

    print(f"model={args.model} device={args.device} max_tokens={args.max_tokens}\n")

    results: list[Result] = []
    reference = ReferenceEngine(config)
    results.append(time_serial("reference", reference, PROMPTS, args.max_tokens))

    paged = PagedEngine(config)
    results.append(time_serial("paged", paged, PROMPTS, args.max_tokens))
    results.append(time_batched(paged, PROMPTS, args.max_tokens))

    speculative = SpeculativeEngine(config, lookahead=args.lookahead)
    results.append(time_serial("speculative", speculative, PROMPTS, args.max_tokens))

    baseline = results[0].tokens_per_second
    for result in results:
        if baseline:
            result.speedup_vs_reference = result.tokens_per_second / baseline

    width = max(len(r.engine) for r in results)
    print(f"{'engine':<{width}}  {'tok/s':>9}  {'speedup':>8}  {'median ms':>10}")
    for result in results:
        print(
            f"{result.engine:<{width}}  "
            f"{result.tokens_per_second:>9.1f}  "
            f"{result.speedup_vs_reference:>7.2f}x  "
            f"{result.median_latency_ms:>10.1f}"
        )
        if result.acceptance_rate:
            print(f"{'':<{width}}  acceptance rate {result.acceptance_rate:.1%}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "device": args.device,
        "max_tokens": args.max_tokens,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "results": [asdict(r) for r in results],
    }
    path = args.out / f"bench-{int(time.time())}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
