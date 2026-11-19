"""Compare original indexed and cached-unfold next-token samplers on CPU.

Runs use identical seeds, batches, lengths, and warmup. No guaranteed
speedup is assumed. Sampling only; no GPT optimizer or disk I/O.
"""

import argparse
import csv
from pathlib import Path

import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.data import sample_batch
from rebuild_gpt.scaling_grid import parse_positive_sizes
from rebuild_gpt.window_sampler import CachedWindowSampler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="4,16")
    parser.add_argument("--contexts", default="4,16")
    parser.add_argument("--corpus-tokens", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=159)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if args.corpus_tokens < 2 or args.iterations < 1 or args.warmup < 0 or args.seed < 0:
        parser.error("corpus-tokens and iterations must be positive; warmup and seed nonnegative")
    if args.csv and args.json and Path(args.csv).resolve() == Path(args.json).resolve():
        parser.error("--csv and --json must be distinct")
    try:
        batch_sizes = parse_positive_sizes(args.batch_sizes, label="batch-sizes")
        contexts = parse_positive_sizes(args.contexts, label="contexts")
        if len(batch_sizes) * len(contexts) > 16:
            raise ValueError("benchmark grid must contain at most 16 shapes")
        if any(context >= args.corpus_tokens for context in contexts):
            raise ValueError("every context must be shorter than corpus-tokens")
    except ValueError as exc:
        parser.error(str(exc))

    tokens = torch.arange(args.corpus_tokens, dtype=torch.long) % 32
    rows = []
    for batch_size in batch_sizes:
        for block_size in contexts:
            sampler = CachedWindowSampler(tokens, block_size=block_size)
            original_rng = torch.Generator().manual_seed(args.seed)
            cached_rng = torch.Generator().manual_seed(args.seed)
            # Verify several consecutive draws, not just one batch.
            for _ in range(3):
                ref = sample_batch(tokens, batch_size=batch_size,
                                   block_size=block_size, generator=original_rng)
                got = sampler.sample(batch_size=batch_size, generator=cached_rng)
                assert all(torch.equal(a, b) for a, b in zip(ref, got))
            assert torch.equal(original_rng.get_state(), cached_rng.get_state())

            def timed_original():
                return sample_batch(tokens, batch_size=batch_size,
                                    block_size=block_size, generator=original_rng)

            def timed_cached():
                return sampler.sample(batch_size=batch_size, generator=cached_rng)

            indexed = time_operation(
                timed_original, device=torch.device("cpu"),
                warmup=args.warmup, iterations=args.iterations,
            )
            cached = time_operation(
                timed_cached, device=torch.device("cpu"),
                warmup=args.warmup, iterations=args.iterations,
            )
            assert torch.equal(original_rng.get_state(), cached_rng.get_state())
            row = {
                "batch_size": batch_size,
                "context": block_size,
                "corpus_tokens": args.corpus_tokens,
                "indexed_median_ms": indexed.median_ms,
                "cached_median_ms": cached.median_ms,
                "indexed_to_cached_latency_ratio": indexed.median_ms / cached.median_ms,
                "unfold_view_storage_bytes": sampler.windows.untyped_storage().nbytes(),
                "corpus_storage_bytes": tokens.untyped_storage().nbytes(),
            }
            rows.append(row)
            print(
                f"B={batch_size} T={block_size} "
                f"indexed={indexed.median_ms:.4f}ms "
                f"cached={cached.median_ms:.4f}ms "
                f"ratio={row['indexed_to_cached_latency_ratio']:.3f}x"
            )
    if args.csv is not None:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", path)
    if args.json is not None:
        report = build_benchmark_report(
            rows=rows, device=torch.device("cpu"),
            settings={
                "batch_sizes": list(batch_sizes),
                "contexts": list(contexts),
                "corpus_tokens": args.corpus_tokens,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "seed": args.seed,
                "benchmarks": ["indexed", "cached"],
                "contains_model_compute": False,
            },
        )
        report["limitations"] = (
            "Sampler-only CPU timing with fixed token stream; excludes model "
            "updates and disk input. Ratio can vary with CPU and workload."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print("All compared token batches and RNG progressions were identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
