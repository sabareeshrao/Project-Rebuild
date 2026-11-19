"""Measure CPU batch sampling and optional end-to-end CPU-to-CUDA copies.

Each timed call includes sampling and (on CUDA) pinning and transfer.
This does NOT include model computation or guarantee transfer overlap.
"""

import argparse
import csv
from pathlib import Path

import torch

from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.data import sample_batch
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.transfers import stage_cpu_batch, transfer_microbatch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=149)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.block_size < 1 or args.iterations < 1
            or args.warmup < 0 or args.seed < 0):
        parser.error("batch, block, iterations must be positive; warmup/seed nonnegative")
    if args.csv and args.json and Path(args.csv).resolve() == Path(args.json).resolve():
        parser.error("--csv and --json must be different paths")
    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(args.seed)
    # A deterministic, contiguous CPU token stream; no disk I/O in timing.
    tokens = torch.arange(8192, dtype=torch.long) % 32
    modes = (
        [("cpu_sampling_only", False, False)]
        if device.type == "cpu" else
        [
            ("pageable_blocking", False, False),
            ("pinned_blocking", True, False),
            ("pinned_nonblocking", True, True),
        ]
    )
    rows = []
    for name, pinned, non_blocking in modes:
        generator = torch.Generator().manual_seed(args.seed)
        def pipeline():
            x, y = sample_batch(
                tokens, batch_size=args.batch_size, block_size=args.block_size,
                generator=generator,
            )
            if device.type == "cuda":
                x, y = stage_cpu_batch(x, y, device=device, pin_memory=pinned)
                gx, gy = transfer_microbatch(
                    x, y, device=device, non_blocking=non_blocking,
                )
                # Complete any asynchronous copies while pinned sources
                # are still alive, before this function returns.
                torch.cuda.synchronize(device)
                return gx, gy
            return x, y

        timing = time_operation(
            pipeline, device=device,
            iterations=args.iterations, warmup=args.warmup,
        )
        row = {
            "mode": name,
            "device": str(device),
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "bytes_per_two_token_batches": 2 * args.batch_size * args.block_size * 8,
            "median_pipeline_ms": timing.median_ms,
            "sample_count": len(timing.samples_ms),
        }
        rows.append(row)
        print(f"{name}: median sampling+staging+copy {timing.median_ms:.4f} ms")
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
            rows=rows, device=device,
            settings={
                "requested_device": args.device,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "seed": args.seed,
                "source": "synthetic_cpu_token_stream",
                "includes_model_compute": False,
            },
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print("CPU reports sampling only; CUDA measures host staging/copy, not compute overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
