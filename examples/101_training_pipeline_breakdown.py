"""Measure sampling, host transfer and GPT optimizer time independently.

CPU reports meaningful sample/update stages but no CUDA transfer.
CUDA variants measure page-locked staging plus transfer time and use
the same current stream: this is intentionally serialized profiling.
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
from rebuild_gpt.training import train_step
from rebuild_gpt.transfers import stage_cpu_batch, transfer_microbatch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1 or args.block_size < 1 or args.n_layer < 1
        or args.n_embd < 1 or args.n_embd % 4 != 0
        or args.iterations < 1 or args.warmup < 0 or args.seed < 0
    ):
        parser.error("batch, context, depth, width and iterations must be positive (width divisible by 4); warmup/seed nonnegative")
    if args.csv and args.json and Path(args.csv).resolve() == Path(args.json).resolve():
        parser.error("--csv and --json must have different paths")
    try:
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    tokens = torch.arange(8192, dtype=torch.long) % 32
    config = GPTConfig(
        vocab_size=32, block_size=args.block_size, n_embd=args.n_embd,
        n_head=4, n_layer=args.n_layer, attention_impl="sdpa",
    )
    modes = (
        [("cpu_only", False, False)] if device.type == "cpu" else [
            ("pageable_blocking", False, False),
            ("pinned_blocking", True, False),
            ("pinned_nonblocking", True, True),
        ]
    )
    rows = []
    for name, pin, non_blocking in modes:
        torch.manual_seed(args.seed)
        model = GPTLanguageModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = make_grad_scaler(device, args.precision)
        generator = torch.Generator().manual_seed(args.seed + 1)

        def sample():
            return sample_batch(
                tokens, batch_size=args.batch_size, block_size=args.block_size,
                generator=generator,
            )

        def transfer(batch):
            x, y = batch
            x, y = stage_cpu_batch(x, y, device=device, pin_memory=pin)
            return transfer_microbatch(x, y, device=device, non_blocking=non_blocking)

        def update(batch):
            return train_step(
                model, optimizer, *batch, precision=args.precision, scaler=scaler,
            )

        timing = time_training_pipeline(
            sample, transfer, update, device=device,
            iterations=args.iterations, warmup=args.warmup,
        )
        tokens_per_update = args.batch_size * args.block_size
        row = {
            "mode": name,
            "device": str(device),
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "tokens_per_update": tokens_per_update,
            "sampling_median_ms": timing.sampling.median_ms,
            "transfer_median_ms": timing.transfer.median_ms,
            "update_median_ms": timing.update.median_ms,
            "end_to_end_median_ms": timing.end_to_end.median_ms,
            "end_to_end_tokens_per_second":
                tokens_per_update * 1000 / timing.end_to_end.median_ms,
            "slowest_stage": timing.slowest_stage,
        }
        rows.append(row)
        print(
            f"{name}: sample={timing.sampling.median_ms:.3f}ms "
            f"transfer={timing.transfer.median_ms:.3f}ms "
            f"update={timing.update.median_ms:.3f}ms "
            f"total={timing.end_to_end.median_ms:.3f}ms "
            f"slowest={timing.slowest_stage}"
        )
    if args.csv is not None:
        target = Path(args.csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", target)
    if args.json is not None:
        report = build_benchmark_report(
            rows=rows, device=device,
            settings={
                "device_requested": args.device,
                "precision": args.precision,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_embd": args.n_embd,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "seed": args.seed,
                "attention_impl": "sdpa",
                "measurement": "sequential_stage_timing_no_overlap",
            },
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print("Stage barriers prevent overlap; these values are not device-independent predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
