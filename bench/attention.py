"""Measure what the paged-attention kernel actually saves.

The kernel exists to remove `PagedKVCache.gather`, whose cost grows with
context length: every decode step copies the whole cached context before
attention can read it. This times both paths across context lengths, so the
claim is a measurement rather than an assertion.

    python bench/attention.py --context 128 512 2048

Requires CUDA; the kernel has no CPU path.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "worker"))

import torch
from tessera_worker.attention import (
    HAS_TRITON,
    paged_attention_torch,
    paged_attention_triton,
)


def build(context: int, num_heads: int, head_dim: int, block_size: int, device: str):
    used = (context + block_size - 1) // block_size
    query = torch.randn(num_heads, head_dim, device=device)
    key_cache = torch.randn(used, num_heads, block_size, head_dim, device=device)
    value_cache = torch.randn(used, num_heads, block_size, head_dim, device=device)
    # Reversed ids, so the benchmark pays the same scattered-read cost a real
    # pool imposes after churn rather than a best-case contiguous walk.
    table = torch.tensor(list(reversed(range(used))), dtype=torch.int32, device=device)
    return query, key_cache, value_cache, table


def timed(fn, *args, warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters  # microseconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "bench" / "results")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs CUDA")
    if not HAS_TRITON:
        raise SystemExit("this benchmark needs Triton")

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"heads={args.heads} head_dim={args.head_dim} block={args.block_size}\n")
    print(f"{'context':>8}  {'gather+dense':>13}  {'kernel':>9}  {'speedup':>8}")

    rows = []
    for context in args.context:
        query, key_cache, value_cache, table = build(
            context, args.heads, args.head_dim, args.block_size, "cuda"
        )
        baseline = timed(
            paged_attention_torch, query, key_cache, value_cache, table, context
        )
        kernel = timed(
            paged_attention_triton, query, key_cache, value_cache, table, context
        )
        speedup = baseline / kernel if kernel else 0.0
        rows.append(
            {
                "context": context,
                "gather_dense_us": baseline,
                "kernel_us": kernel,
                "speedup": speedup,
            }
        )
        print(f"{context:>8}  {baseline:>12.1f}u  {kernel:>8.1f}u  {speedup:>7.2f}x")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"attention-{int(time.time())}.json"
    path.write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "heads": args.heads,
                "head_dim": args.head_dim,
                "block_size": args.block_size,
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
