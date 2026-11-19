"""Compare three identical sampled-validation strategies and their RAM cost.

This experiment times evaluation only; optimizer updates and disk
loading are not included. Index plans reference the existing CPU
corpus, which is excluded from retained plan bytes.
"""

import argparse
import csv
from pathlib import Path

import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.eval_cache import prepare_train_val_evaluation_batches
from rebuild_gpt.eval_index_plan import prepare_train_val_index_plans
from rebuild_gpt.evaluation import (
    estimate_train_val_loss, evaluate_prepared_train_val_loss,
    evaluate_index_plan_train_val_loss,
)
from rebuild_gpt.gpt import GPTLanguageModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=177)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1 or args.context < 1
        or args.eval_batches < 1 or args.iterations < 1
        or args.warmup < 0 or args.seed < 0
    ):
        parser.error("batch-size, context, eval-batches and iterations must be positive; warmup/seed nonnegative")
    if args.csv and args.json and Path(args.csv).resolve() == Path(args.json).resolve():
        parser.error("--csv and --json must use different paths")
    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(args.seed)
    model = GPTLanguageModel(GPTConfig(
        vocab_size=32, block_size=args.context,
        n_embd=16, n_head=4, n_layer=1,
        attention_impl="sdpa", dropout=0.0,
    )).to(device).train()
    train = torch.arange(512, dtype=torch.long) % 32
    val = (torch.arange(320, dtype=torch.long) * 3) % 32
    common = dict(
        batch_size=args.batch_size, block_size=args.context,
        eval_batches=args.eval_batches, seed=args.seed + 2,
    )
    full_plans = prepare_train_val_evaluation_batches(train, val, **common)
    index_plans = prepare_train_val_index_plans(train, val, **common)

    implementations = (
        ("regenerate", lambda: estimate_train_val_loss(
            model, train, val, **common,
        ), 0),
        ("cached", lambda: evaluate_prepared_train_val_loss(
            model, *full_plans,
        ), sum(plan.storage_bytes for plan in full_plans)),
        ("indices", lambda: evaluate_index_plan_train_val_loss(
            model, *index_plans,
        ), sum(plan.index_storage_bytes for plan in index_plans)),
    )
    expected = implementations[0][1]()
    rows = []
    for name, fn, bytes_retained in implementations:
        actual = fn()
        assert actual == expected, f"{name} changed sampled evaluation scores"
        timing = time_operation(
            fn, device=device, iterations=args.iterations, warmup=args.warmup,
        )
        rows.append({
            "backend": name,
            "device": str(device),
            "batch_size": args.batch_size,
            "context": args.context,
            "eval_batches_per_split": args.eval_batches,
            "retained_plan_bytes": bytes_retained,
            "median_evaluation_ms": timing.median_ms,
            "validation_loss": actual["validation_loss"],
        })
        print(
            f"{name}: median={timing.median_ms:.4f} ms "
            f"retained_plan={bytes_retained} bytes"
        )
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", path)
    if args.json:
        report = build_benchmark_report(
            rows=rows, device=device, settings={
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batches": args.eval_batches,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "seed": args.seed,
                "includes_optimizer_updates": False,
                "includes_corpus_storage": False,
            },
        )
        report["limitations"] = (
            "Evaluations only. Index-only plans retain references to the CPU "
            "corpus and gather windows on each evaluation. Timing excludes "
            "preparation and does not guarantee speedups on other hardware."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print("All three backends produced identical validation scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
