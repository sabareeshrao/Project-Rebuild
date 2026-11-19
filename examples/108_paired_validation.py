"""Compare GPT before/after one training step on paired validation windows.

A single synthetic-data optimizer update does not guarantee improved
held-out loss; the same fixed seeded windows are used for both models.
"""

import argparse
import copy
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.data import load_corpus, sample_batch, split_train_val
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.paired_evaluation import compare_paired_validation
from rebuild_gpt.training import train_step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--first-seed", type=int, default=182)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--model-seed", type=int, default=71)
    parser.add_argument("--json", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)
    if (any(getattr(args, field) < 1 for field in (
        "batch_size", "context", "eval_batches", "seed_count",
    )) or args.first_seed < 0 or args.model_seed < 0):
        parser.error("batch, context, eval-batches and seed-count must be positive; seeds nonnegative")
    if (args.csv and args.json and Path(args.csv).resolve() == Path(args.json).resolve()):
        parser.error("--csv and --json must use different paths")
    try:
        device = resolve_device(args.device)
        tokens, tokenizer = load_corpus(args.corpus)
        train, val = split_train_val(tokens, val_fraction=0.2)
        if len(val) <= args.context:
            raise ValueError("validation split is too short for context")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    torch.manual_seed(args.model_seed)
    baseline = GPTLanguageModel(GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.context,
        n_embd=16, n_head=4, n_layer=1, attention_impl="sdpa",
    )).to(device)
    updated = copy.deepcopy(baseline)
    optimizer = torch.optim.AdamW(updated.parameters(), lr=0.001)
    train_rng = torch.Generator(device="cpu").manual_seed(args.model_seed + 1)
    x, y = sample_batch(
        train, batch_size=args.batch_size,
        block_size=args.context, generator=train_rng,
    )
    step_loss = train_step(updated, optimizer, x.to(device), y.to(device))
    chosen = tuple(range(args.first_seed, args.first_seed + args.seed_count))
    result = compare_paired_validation(
        baseline, updated, val, seeds=chosen,
        batch_size=args.batch_size, block_size=args.context,
        eval_batches=args.eval_batches,
    )
    rows = [
        {
            "seed": seed, "baseline_loss": a,
            "after_one_update_loss": b, "delta_after_minus_before": d,
        }
        for seed, a, b, d in zip(
            result.seeds, result.baseline_means,
            result.candidate_means, result.candidate_minus_baseline,
        )
    ]
    for row in rows:
        print(
            f"seed={row['seed']} baseline={row['baseline_loss']:.6f} "
            f"updated={row['after_one_update_loss']:.6f} "
            f"delta={row['delta_after_minus_before']:+.6f}"
        )
    print(
        f"mean paired change={result.delta_summary.mean:+.6f}; "
        f"one training-step loss={step_loss:.6f}"
    )
    if args.csv:
        target = Path(args.csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", target)
    if args.json:
        report = build_benchmark_report(
            rows=rows, device=device, settings={
                "corpus": str(args.corpus),
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batches_per_seed": args.eval_batches,
                "first_seed": args.first_seed,
                "seed_count": args.seed_count,
                "model_seed": args.model_seed,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "optimizer_steps_for_candidate": 1,
                "training_step_loss": step_loss,
            },
        )
        report["paired_delta_summary"] = asdict(result.delta_summary)
        report["limitations"] = (
            "A negative paired delta indicates lower loss on the sampled "
            "held-out windows only; this is not a significance test or a "
            "claim of generalized improvement."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
