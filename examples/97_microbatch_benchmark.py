"""Benchmark equal logical batches split into different microbatch sizes.

Timing covers GPT forward, backward, and AdamW, but not data loading.
GPU allocator peak deltas are reported only on actual CUDA hardware.
"""

import argparse
import csv
import gc
from pathlib import Path

import torch

from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.cuda_memory import measure_cuda_peak
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.memory import mebibytes
from rebuild_gpt.microbatches import microbatch_sizes
from rebuild_gpt.precision import make_grad_scaler, resolve_precision
from rebuild_gpt.scaling_grid import parse_positive_sizes
from rebuild_gpt.training import train_logical_batch_step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--attention-impl", choices=("fused", "sdpa"), default="sdpa")
    parser.add_argument("--logical-batch-size", type=int, default=6)
    parser.add_argument("--microbatch-sizes", default="1,2,3,6")
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=139)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.logical_batch_size < 1 or args.context < 1
        or args.n_embd < 1 or args.n_head < 1 or args.n_layer < 1
        or args.iterations < 1 or args.warmup < 0 or args.seed < 0
    ):
        parser.error("batch, context, model dimensions and iterations must be positive; warmup/seed nonnegative")
    if args.n_embd % args.n_head != 0:
        parser.error("n-embd must be divisible by n-head")
    if (args.csv is not None and args.json is not None
            and Path(args.csv).resolve() == Path(args.json).resolve()):
        parser.error("--csv and --json must use distinct paths")
    try:
        sizes = parse_positive_sizes(args.microbatch_sizes, label="microbatch-sizes")
        if any(size > args.logical_batch_size for size in sizes):
            raise ValueError("microbatch size cannot exceed logical batch size")
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    config = GPTConfig(
        vocab_size=32, block_size=args.context,
        n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer,
        attention_impl=args.attention_impl,
    )
    torch.manual_seed(args.seed + 1)
    inputs = torch.randint(
        0, config.vocab_size,
        (args.logical_batch_size, args.context), device=device,
    )
    targets = torch.randint(
        0, config.vocab_size,
        (args.logical_batch_size, args.context), device=device,
    )

    rows = []
    for micro_size in sizes:
        torch.manual_seed(args.seed)
        model = GPTLanguageModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = make_grad_scaler(device, args.precision)

        def step():
            return train_logical_batch_step(
                model, optimizer, inputs, targets,
                microbatch_size=micro_size,
                precision=args.precision, scaler=scaler,
            )

        timing = time_operation(
            step, device=device, warmup=args.warmup,
            iterations=args.iterations,
        )
        delta_peak_mib = None
        if device.type == "cuda":
            # Optimizer state is initialized during timed training;
            # this next step measures extra allocations above that baseline.
            delta_peak_mib = mebibytes(
                measure_cuda_peak(step, device=device).additional_peak_allocated_bytes
            )
        tokens = args.logical_batch_size * args.context
        row = {
            "logical_batch_size": args.logical_batch_size,
            "microbatch_size": micro_size,
            "chunks_per_step": len(microbatch_sizes(args.logical_batch_size, micro_size)),
            "context": args.context,
            "tokens_per_update": tokens,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "median_step_ms": timing.median_ms,
            "tokens_per_second": tokens * 1000.0 / timing.median_ms,
            "additional_cuda_peak_allocated_mib": delta_peak_mib,
        }
        rows.append(row)
        print(
            f"logical={args.logical_batch_size} micro={micro_size} "
            f"chunks={row['chunks_per_step']} "
            f"median={timing.median_ms:.3f}ms "
            f"tokens/s={row['tokens_per_second']:.1f} "
            f"CUDA peak delta={'n/a' if delta_peak_mib is None else f'{delta_peak_mib:.4f} MiB'}"
        )
        del model, optimizer, scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.csv is not None:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", output)
    if args.json is not None:
        report = build_benchmark_report(
            rows=rows, device=device,
            settings={
                "device_requested": args.device,
                "precision": args.precision,
                "attention_impl": args.attention_impl,
                "logical_batch_size": args.logical_batch_size,
                "microbatch_sizes": list(sizes),
                "context": args.context,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "seed": args.seed,
            },
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print("Measurements exclude data loading; CUDA peak delta is not total VRAM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
