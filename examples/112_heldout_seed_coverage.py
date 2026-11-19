"""Measure token-window coverage on a real UTF-8 corpus's held-out split.

This is a sampler-only diagnostic: no model forward or optimizer step
occurs, and repeated positions are not independent observations.
"""

import argparse
import csv
from dataclasses import asdict
import math
from pathlib import Path

import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.coverage_growth import analyze_coverage_growth
from rebuild_gpt.coverage_targets import assess_coverage_target
from rebuild_gpt.cross_seed_overlap import analyze_cross_seed_overlap
from rebuild_gpt.data import load_corpus, split_train_val
from rebuild_gpt.disjoint_evaluation import prepare_disjoint_evaluation_index_plan
from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan
from rebuild_gpt.scaling_grid import parse_positive_sizes
from rebuild_gpt.shifted_evaluation import prepare_shifted_evaluation_index_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--seed-counts", default="1,2,4")
    parser.add_argument("--first-seed", type=int, default=208)
    parser.add_argument(
        "--sampling-policy",
        choices=("replacement", "disjoint", "shifted", "all"),
        default="all",
    )
    parser.add_argument(
        "--coverage-target", type=float, default=None,
        help="Optional retrospective fraction of held-out positions in (0,1]",
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1 or args.context < 1
        or args.eval_batches < 1 or args.first_seed < 0
    ):
        parser.error("batch-size, context and eval-batches must be positive; first-seed nonnegative")
    source = Path(args.corpus).resolve()
    outputs = [Path(p).resolve() for p in (args.csv, args.json) if p is not None]
    if len(set(outputs)) != len(outputs) or source in outputs:
        parser.error("CSV and JSON outputs must differ and cannot overwrite corpus")
    try:
        counts = parse_positive_sizes(args.seed_counts, label="seed-counts")
        if len(counts) > 8 or max(counts) > 32:
            raise ValueError("seed-counts supports at most eight sizes, each at most 32")
        if args.coverage_target is not None and (
            not math.isfinite(args.coverage_target)
            or not 0 < args.coverage_target <= 1
        ):
            raise ValueError("coverage-target must be finite and in (0,1]")
        corpus_tokens, tokenizer = load_corpus(args.corpus)
        _, heldout = split_train_val(corpus_tokens, val_fraction=0.2)
        if heldout.numel() <= args.context:
            raise ValueError("held-out split must exceed the requested context")
        builders = {
            "replacement": prepare_evaluation_index_plan,
            "disjoint": prepare_disjoint_evaluation_index_plan,
            "shifted": prepare_shifted_evaluation_index_plan,
        }
        policies = (
            ("replacement", "disjoint", "shifted")
            if args.sampling_policy == "all" else (args.sampling_policy,)
        )
        rows, traces = [], []
        for policy in policies:
            for count in counts:
                plans = tuple(
                    builders[policy](
                        heldout, batch_size=args.batch_size,
                        block_size=args.context,
                        eval_batches=args.eval_batches,
                        seed=args.first_seed + index,
                    )
                    for index in range(count)
                )
                union = analyze_cross_seed_overlap(plans)
                growth = analyze_coverage_growth(plans)
                row = {
                    "sampling_policy": policy,
                    "seed_count": count,
                    "heldout_token_count": heldout.numel(),
                    "global_unique_corpus_token_positions":
                        union.unique_corpus_token_positions,
                    "global_corpus_coverage_fraction": union.coverage_fraction,
                    "last_seed_new_token_positions":
                        growth[-1].new_corpus_token_positions,
                    "repeated_positions_between_seeds":
                        union.repeated_positions_between_seeds,
                    "repeated_positions_within_seeds":
                        union.repeated_positions_within_seeds,
                }
                if args.coverage_target is not None:
                    target = assess_coverage_target(
                        growth, corpus_token_count=heldout.numel(),
                        target_fraction=args.coverage_target,
                    )
                    row.update({
                        "coverage_target_reached": target.reached,
                        "first_reaching_seed_number": target.first_reaching_seed_number,
                        "additional_positions_needed": target.additional_positions_needed,
                    })
                rows.append(row)
                traces.append({
                    "sampling_policy": policy,
                    "seed_count": count,
                    "steps": [asdict(step) for step in growth],
                })
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    print(
        f"UTF-8 corpus chars={corpus_tokens.numel()}, "
        f"vocab={tokenizer.vocab_size}, held-out tokens={heldout.numel()}"
    )
    for row in rows:
        print(
            f"policy={row['sampling_policy']} seeds={row['seed_count']} "
            f"unique={row['global_unique_corpus_token_positions']} "
            f"coverage={row['global_corpus_coverage_fraction']:.3f}"
        )
    if args.csv is not None:
        destination = Path(args.csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", destination)
    if args.json is not None:
        report = build_benchmark_report(
            rows=rows, device=torch.device("cpu"), settings={
                "corpus": str(args.corpus),
                "corpus_token_count": corpus_tokens.numel(),
                "vocabulary_size": tokenizer.vocab_size,
                "heldout_token_count": heldout.numel(),
                "heldout_fraction": 0.2,
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batches_per_seed": args.eval_batches,
                "first_seed": args.first_seed,
                "seed_counts": list(counts),
                "sampling_policy": args.sampling_policy,
                "coverage_target_fraction": args.coverage_target,
                "contains_model_compute": False,
                "corpus_is_synthetic": False,
            },
        )
        report["marginal_coverage"] = traces
        report["limitations"] = (
            "Sampler-only diagnostics on a contiguous held-out portion of "
            "one real UTF-8 corpus. No GPT forward or optimizer update is "
            "performed. Across-seed windows can overlap even for disjoint "
            "policies. Token coverage does not establish statistical "
            "independence, calibration, or generalization performance. "
            "Optional coverage targets are retrospective, not stopping rules."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
