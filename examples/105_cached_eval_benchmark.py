"""Compare repeated validation with and without precomputed CPU batches.

One-time plan preparation is reported separately; no model updates
occur during the experiment. Results depend on hardware and shape.
"""

import argparse
import csv
import math
from pathlib import Path

import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.cache_economics import minimum_break_even_evaluations, estimated_net_savings_ms
from rebuild_gpt.eval_cache_budget import (
    estimate_evaluation_cache_budget, enforce_evaluation_cache_budget,
)
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.eval_cache import prepare_train_val_evaluation_batches
from rebuild_gpt.evaluation import (
    estimate_train_val_loss, evaluate_prepared_train_val_loss,
)
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.scaling_grid import parse_positive_sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--eval-batches", default="1,2,4")
    parser.add_argument("--eval-cache-max-mib", type=float, default=None,
                        help="Reject requested shapes exceeding this combined CPU cache budget")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=164)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.context < 1 or args.warmup < 0
            or args.iterations < 1 or args.seed < 0):
        parser.error("batch-size, context, iterations must be positive; warmup/seed nonnegative")
    if args.csv and args.json and Path(args.csv).resolve() == Path(args.json).resolve():
        parser.error("--csv and --json must use distinct paths")
    try:
        counts = parse_positive_sizes(args.eval_batches, label="eval-batches")
        device = resolve_device(args.device)
        max_bytes = None
        if args.eval_cache_max_mib is not None:
            if not math.isfinite(args.eval_cache_max_mib) or args.eval_cache_max_mib <= 0:
                raise ValueError("eval-cache-max-mib must be finite and positive")
            max_bytes = int(args.eval_cache_max_mib * (1024 * 1024))
            if max_bytes < 1:
                raise ValueError("eval-cache-max-mib must permit at least one byte")
        # Check EVERY shape before model construction or any batch sampling.
        for count in counts:
            estimate = estimate_evaluation_cache_budget(
                batch_size=args.batch_size, block_size=args.context,
                eval_batches=count,
            )
            enforce_evaluation_cache_budget(estimate.total_bytes, max_bytes)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(args.seed)
    config = GPTConfig(
        vocab_size=32, block_size=args.context, n_embd=16,
        n_head=4, n_layer=1, attention_impl="sdpa", dropout=0.15,
    )
    model = GPTLanguageModel(config).to(device).train()
    train = torch.arange(512, dtype=torch.long) % 32
    val = (torch.arange(320, dtype=torch.long) * 7) % 32

    rows = []
    for count in counts:
        def prepare():
            return prepare_train_val_evaluation_batches(
                train, val, batch_size=args.batch_size,
                block_size=args.context, eval_batches=count,
                seed=args.seed + 2, max_bytes=max_bytes,
            )

        timing_prepare = time_operation(
            prepare, device=device,
            warmup=args.warmup, iterations=args.iterations,
        )
        cached_train, cached_val = prepare()
        before = torch.get_rng_state().clone()
        reference = estimate_train_val_loss(
            model, train, val, batch_size=args.batch_size,
            block_size=args.context, eval_batches=count, seed=args.seed + 2,
        )
        cached = evaluate_prepared_train_val_loss(model, cached_train, cached_val)
        assert reference == cached, "cached batches changed evaluation losses"
        assert torch.equal(before, torch.get_rng_state())
        assert model.training

        original_time = time_operation(
            lambda: estimate_train_val_loss(
                model, train, val, batch_size=args.batch_size,
                block_size=args.context, eval_batches=count, seed=args.seed + 2,
            ),
            device=device, warmup=args.warmup, iterations=args.iterations,
        )
        cached_time = time_operation(
            lambda: evaluate_prepared_train_val_loss(model, cached_train, cached_val),
            device=device, warmup=args.warmup, iterations=args.iterations,
        )
        break_even = minimum_break_even_evaluations(
            preparation_ms=timing_prepare.median_ms,
            regenerated_ms=original_time.median_ms,
            cached_ms=cached_time.median_ms,
        )
        row = {
            "device": str(device),
            "cache_budget_mib": args.eval_cache_max_mib,
            "break_even_evaluation_events": break_even,
            "estimated_net_savings_10_events_ms": estimated_net_savings_ms(
                preparation_ms=timing_prepare.median_ms,
                regenerated_ms=original_time.median_ms,
                cached_ms=cached_time.median_ms,
                evaluation_events=10,
            ),
            "batch_size": args.batch_size,
            "context": args.context,
            "eval_batches_per_split": count,
            "plan_storage_bytes": cached_train.storage_bytes + cached_val.storage_bytes,
            "plan_preparation_median_ms": timing_prepare.median_ms,
            "regenerated_eval_median_ms": original_time.median_ms,
            "cached_eval_median_ms": cached_time.median_ms,
            "regenerated_to_cached_latency_ratio":
                original_time.median_ms / cached_time.median_ms,
            "validation_loss": reference["validation_loss"],
        }
        rows.append(row)
        print(
            f"eval_batches={count} prepare={timing_prepare.median_ms:.3f}ms "
            f"regenerate={original_time.median_ms:.3f}ms "
            f"cached={cached_time.median_ms:.3f}ms "
            f"CPU_cache={row['plan_storage_bytes']} bytes "
            f"break_even={break_even if break_even is not None else 'not reached'}"
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
            rows=rows, device=device,
            settings={
                "device_requested": args.device,
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batches": list(counts),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "seed": args.seed,
                "measurement": "evaluation_only_no_optimizer",
                "preparation_reported_separately": True,
                "eval_cache_max_mib": args.eval_cache_max_mib,
                "break_even_is_estimate": True,
            },
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print("The cache trades host RAM for CPU sampling work; speedup is not guaranteed.")
    print("Break-even calculations use measured medians, not guaranteed future savings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
