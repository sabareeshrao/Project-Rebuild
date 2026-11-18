"""Sweep context lengths, recording manual vs SDPA inference timing as CSV."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch

from rebuild_gpt.benchmark_sweep import parse_context_lengths
from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.precision import autocast_context, resolve_precision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--lengths", default="4,8,16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("batch-size and iterations must be positive; warmup nonnegative")
    try:
        lengths = parse_context_lengths(args.lengths)
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    rows = []
    for context in lengths:
        torch.manual_seed(119)
        config = GPTConfig(
            vocab_size=64, block_size=context, n_embd=32,
            n_head=4, n_layer=2, attention_impl="fused",
        )
        manual = GPTLanguageModel(config).to(device).eval()
        sdpa = GPTLanguageModel(replace(config, attention_impl="sdpa")).to(device).eval()
        sdpa.load_state_dict(manual.state_dict(), strict=True)
        tokens = torch.randint(
            config.vocab_size, (args.batch_size, context), device=device,
        )

        def forward(model):
            with autocast_context(device, args.precision):
                return model(tokens)

        with torch.inference_mode():
            difference = (forward(manual) - forward(sdpa)).abs().max().item()
            manual_ms = time_operation(
                lambda: forward(manual), device=device,
                warmup=args.warmup, iterations=args.iterations,
            ).median_ms
            sdpa_ms = time_operation(
                lambda: forward(sdpa), device=device,
                warmup=args.warmup, iterations=args.iterations,
            ).median_ms

        tokens_per_batch = args.batch_size * context
        rows.append({
            "context": context,
            "batch": args.batch_size,
            "manual_median_ms": manual_ms,
            "sdpa_median_ms": sdpa_ms,
            "manual_tokens_per_second": tokens_per_batch * 1000.0 / manual_ms,
            "sdpa_tokens_per_second": tokens_per_batch * 1000.0 / sdpa_ms,
            "max_abs_logit_difference": difference,
        })
        print(f"context={context:>3} manual={manual_ms:.3f}ms "
              f"sdpa={sdpa_ms:.3f}ms max_err={difference:.3g}")

    if args.csv is not None:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
