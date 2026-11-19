"""Inspect sampled held-out loss sensitivity for an untrained tiny GPT.

This is a diagnostic experiment, not a claim about trained model
quality or a calibrated generalization confidence interval.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.data import load_corpus, split_train_val
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.seed_sensitivity import evaluate_seed_sensitivity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=181)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--json", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)
    if any(getattr(args, field) < 1 for field in (
        "batch_size", "context", "eval_batches", "seed_count",
    )) or args.first_seed < 0 or args.model_seed < 0:
        parser.error("batch, context, eval-batches, seed-count must be positive; seeds nonnegative")
    if (args.json is not None and args.csv is not None
            and Path(args.json).resolve() == Path(args.csv).resolve()):
        parser.error("--json and --csv must be different paths")
    try:
        device = resolve_device(args.device)
        tokens, tokenizer = load_corpus(args.corpus)
        _, val = split_train_val(tokens, val_fraction=0.2)
        if len(val) <= args.context:
            raise ValueError("validation split is too short for context")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    torch.manual_seed(args.model_seed)
    model = GPTLanguageModel(GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.context,
        n_embd=16, n_head=4, n_layer=1,
        attention_impl="sdpa",
    )).to(device).eval()
    chosen = tuple(range(args.first_seed, args.first_seed + args.seed_count))
    data = evaluate_seed_sensitivity(
        model, val, seeds=chosen, batch_size=args.batch_size,
        block_size=args.context, eval_batches=args.eval_batches,
    )
    rows = [
        {"seed": seed, "validation_mean_loss": loss}
        for seed, loss in zip(data.seeds, data.per_seed_means)
    ]
    print(
        f"validation loss across {len(rows)} seeds: "
        f"mean={data.summary.mean:.6f}, "
        f"sample_stdev={data.summary.sample_stdev}"
    )
    for row in rows:
        print(f"seed={row['seed']} loss={row['validation_mean_loss']:.6f}")

    if args.csv is not None:
        target = Path(args.csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["seed", "validation_mean_loss"])
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", target)
    if args.json is not None:
        report = build_benchmark_report(
            rows=rows, device=device,
            settings={
                "corpus": str(args.corpus),
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batches_per_seed": args.eval_batches,
                "first_seed": args.first_seed,
                "seed_count": args.seed_count,
                "model_seed": args.model_seed,
                "model_is_trained": False,
                "validation_split_fraction": 0.2,
            },
        )
        report["summary"] = asdict(data.summary)
        report["limitations"] = (
            "Repeated samples from one held-out corpus may overlap. "
            "Reported standard error is descriptive, not a calibrated "
            "generalization confidence interval. The model is untrained."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
