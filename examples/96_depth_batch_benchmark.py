"""Benchmark complete GPT training-step throughput across batch size and depth.

Synthetic batches are prepared before timing; no data-loading costs.
CPU and CUDA results depend on hardware, precision, and workload.
"""

import argparse
import csv
from pathlib import Path

import torch

from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.precision import make_grad_scaler, resolve_precision
from rebuild_gpt.scaling_grid import build_scaling_grid, parse_positive_sizes
from rebuild_gpt.training import train_accumulated_step, train_step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--attention-impl", choices=("fused", "sdpa"), default="sdpa")
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--depths", default="1,2")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup < 0 or args.n_embd < 1:
        parser.error("iterations and n-embd must be positive; warmup nonnegative")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    try:
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
        grid = build_scaling_grid(
            parse_positive_sizes(args.batch_sizes, label="batch-sizes"),
            (args.context,),
            parse_positive_sizes(args.depths, label="depths"),
            n_head=args.n_head,
            gradient_accumulation_steps=args.accumulation,
            dtype=dtype[args.precision], max_points=16,
        )
        if args.n_embd % args.n_head != 0:
            raise ValueError("n-embd must be divisible by n-head")
    except ValueError as exc:
        parser.error(str(exc))

    rows = []
    for item in grid:
        torch.manual_seed(134)  # same initial weights for matching architectures
        cfg = GPTConfig(
            vocab_size=32, block_size=item.context_length,
            n_embd=args.n_embd, n_head=item.n_head, n_layer=item.n_layer,
            attention_impl=args.attention_impl,
        )
        model = GPTLanguageModel(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = make_grad_scaler(device, args.precision)
        # Fixed synthetic microbatches keep the timed region focused on
        # forward, backward and optimizer rather than input generation.
        microbatches = [
            (
                torch.randint(0, cfg.vocab_size,
                              (item.batch_size, item.context_length), device=device),
                torch.randint(0, cfg.vocab_size,
                              (item.batch_size, item.context_length), device=device),
            )
            for _ in range(item.gradient_accumulation_steps)
        ]
        def step():
            if len(microbatches) == 1:
                return train_step(
                    model, optimizer, *microbatches[0],
                    precision=args.precision, scaler=scaler,
                )
            return train_accumulated_step(
                model, optimizer, microbatches,
                precision=args.precision, scaler=scaler,
            )

        median_ms = time_operation(
            step, device=device, warmup=args.warmup,
            iterations=args.iterations,
        ).median_ms
        tokens_per_second = item.tokens_per_optimizer_step * 1000.0 / median_ms
        row = {
            "device": str(device),
            "precision": args.precision,
            "attention_impl": args.attention_impl,
            "batch_size": item.batch_size,
            "context_length": item.context_length,
            "n_layer": item.n_layer,
            "accumulation": item.gradient_accumulation_steps,
            "tokens_per_update": item.tokens_per_optimizer_step,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "median_step_ms": median_ms,
            "tokens_per_second": tokens_per_second,
        }
        rows.append(row)
        print(
            f"B={item.batch_size} T={item.context_length} L={item.n_layer} "
            f"accum={item.gradient_accumulation_steps} "
            f"median={median_ms:.3f}ms tokens/s={tokens_per_second:.1f}"
        )

    if args.csv is not None:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", path)
    print("Measured step timing excludes data loading; no speedup is guaranteed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
