"""Export analytical GPT workload scaling estimates as CSV (no GPU needed).

This script estimates explicit attention-score storage, not peak VRAM.
Run from the repository root after installing the package.
"""

import argparse
import csv
from pathlib import Path

import torch

from rebuild_gpt.memory import mebibytes
from rebuild_gpt.scaling_grid import build_scaling_grid, parse_positive_sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--contexts", default="4,8")
    parser.add_argument("--depths", default="1,2")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--max-points", type=int, default=64)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)
    dtypes = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    try:
        grid = build_scaling_grid(
            parse_positive_sizes(args.batch_sizes, label="batch-sizes"),
            parse_positive_sizes(args.contexts, label="contexts"),
            parse_positive_sizes(args.depths, label="depths"),
            n_head=args.heads,
            gradient_accumulation_steps=args.accumulation,
            dtype=dtypes[args.dtype],
            max_points=args.max_points,
        )
    except ValueError as exc:
        parser.error(str(exc))

    rows = []
    for item in grid:
        row = {
            "batch_size": item.batch_size,
            "context_length": item.context_length,
            "n_head": item.n_head,
            "n_layer": item.n_layer,
            "accumulation": item.gradient_accumulation_steps,
            "dtype": args.dtype,
            "tokens_per_microbatch": item.tokens_per_microbatch,
            "tokens_per_optimizer_step": item.tokens_per_optimizer_step,
            "explicit_score_bytes_per_layer": item.score_bytes_per_layer,
            "sum_explicit_score_bytes_all_layers": item.sum_score_bytes_all_layers,
        }
        rows.append(row)
        print(
            f"B={item.batch_size} T={item.context_length} L={item.n_layer} "
            f"tokens/update={item.tokens_per_optimizer_step} "
            f"one-score={mebibytes(item.score_bytes_per_layer):.5f} MiB "
            f"all-score-sum={mebibytes(item.sum_score_bytes_all_layers):.5f} MiB"
        )

    if args.csv is not None:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", output)
    print("These score sizes are analytical, not allocated/peak GPU memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
