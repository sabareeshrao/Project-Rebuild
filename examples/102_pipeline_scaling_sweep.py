"""Compare pipeline-stage medians across small GPT batch/depth configurations.

Not a speedup assertion: CPU/GPU timings depend on runtime and hardware.
Instrumenting each stage inserts synchronization barriers on CUDA.
"""

import argparse
import csv
from pathlib import Path

import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.data import sample_batch
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.pipeline_timing import time_training_pipeline
from rebuild_gpt.precision import make_grad_scaler, resolve_precision
from rebuild_gpt.scaling_grid import build_scaling_grid, parse_positive_sizes
from rebuild_gpt.training import train_step
from rebuild_gpt.transfers import transfer_microbatch
from rebuild_gpt.window_sampler import CachedWindowSampler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--sampling-backend", choices=("indexed", "cached"), default="indexed")
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--depths", default="1,2")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup < 0 or args.seed < 0:
        parser.error("iterations must be positive; warmup and seed nonnegative")
    if args.csv and args.json and Path(args.csv).resolve() == Path(args.json).resolve():
        parser.error("--csv and --json must use different paths")
    try:
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
        grid = build_scaling_grid(
            parse_positive_sizes(args.batch_sizes, label="batch-sizes"),
            (args.context,),
            parse_positive_sizes(args.depths, label="depths"),
            n_head=2, max_points=12,
        )
    except ValueError as exc:
        parser.error(str(exc))

    tokens = torch.arange(8192, dtype=torch.long) % 32
    rows = []
    for item in grid:
        torch.manual_seed(args.seed)
        config = GPTConfig(
            vocab_size=32, block_size=item.context_length, n_embd=8,
            n_head=2, n_layer=item.n_layer, attention_impl="sdpa",
        )
        model = GPTLanguageModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = make_grad_scaler(device, args.precision)
        rng = torch.Generator().manual_seed(args.seed + 1)
        sampler = (
            CachedWindowSampler(tokens, block_size=item.context_length)
            if args.sampling_backend == "cached" else None
        )

        def sample():
            if sampler is not None:
                return sampler.sample(
                    batch_size=item.batch_size, generator=rng,
                )
            return sample_batch(
                tokens, batch_size=item.batch_size,
                block_size=item.context_length, generator=rng,
            )

        def transfer(batch):
            return transfer_microbatch(*batch, device=device)

        def update(batch):
            return train_step(
                model, optimizer, *batch,
                precision=args.precision, scaler=scaler,
            )

        timing = time_training_pipeline(
            sample, transfer, update, device=device,
            warmup=args.warmup, iterations=args.iterations,
        )
        row = {
            "device": str(device),
            "sampling_backend": args.sampling_backend,
            "precision": args.precision,
            "batch_size": item.batch_size,
            "context": item.context_length,
            "depth": item.n_layer,
            "tokens_per_update": item.tokens_per_optimizer_step,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "sampling_median_ms": timing.sampling.median_ms,
            "transfer_median_ms": timing.transfer.median_ms,
            "update_median_ms": timing.update.median_ms,
            "end_to_end_median_ms": timing.end_to_end.median_ms,
            "slowest_stage": timing.slowest_stage,
        }
        rows.append(row)
        print(
            f"B={item.batch_size} L={item.n_layer} "
            f"sample={timing.sampling.median_ms:.3f}ms "
            f"transfer={timing.transfer.median_ms:.3f}ms "
            f"update={timing.update.median_ms:.3f}ms "
            f"total={timing.end_to_end.median_ms:.3f}ms "
            f"slowest={timing.slowest_stage}"
        )
    if args.csv is not None:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", path)
    if args.json is not None:
        report = build_benchmark_report(
            rows=rows, device=device,
            settings={
                "device_requested": args.device,
                "precision": args.precision,
                "sampling_backend": args.sampling_backend,
                "batch_sizes": list(dict.fromkeys(item.batch_size for item in grid)),
                "depths": list(dict.fromkeys(item.n_layer for item in grid)),
                "context": args.context,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "seed": args.seed,
                "stage_sync": True,
                "includes_file_io": False,
                "includes_validation": False,
            },
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print("Stage timings contain synchronization barriers and are not hardware-independent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
