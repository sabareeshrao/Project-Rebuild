"""Compare global token coverage as the number of validation seeds grows.

Sampler-only CPU experiment. Plans draw from one shared token stream;
coverage is descriptive, not an effective sample size or confidence bound.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.cross_seed_overlap import analyze_cross_seed_overlap
from rebuild_gpt.coverage_growth import analyze_coverage_growth
from rebuild_gpt.disjoint_evaluation import prepare_disjoint_evaluation_index_plan
from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan
from rebuild_gpt.scaling_grid import parse_positive_sizes
from rebuild_gpt.shifted_evaluation import prepare_shifted_evaluation_index_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--seed-counts", default="1,2,4")
    parser.add_argument("--first-seed", type=int, default=203)
    parser.add_argument(
        "--sampling-policy",
        choices=("replacement", "disjoint", "shifted", "all"),
        default="all",
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.corpus_tokens < 2 or args.batch_size < 1
        or args.context < 1 or args.eval_batches < 1
        or args.first_seed < 0
    ):
        parser.error("corpus-tokens, batch-size, context, eval-batches must be positive; first-seed nonnegative")
    sources = [Path(p).resolve() for p in (args.csv, args.json) if p is not None]
    if len(set(sources)) != len(sources):
        parser.error("--csv and --json must be different paths")
    try:
        counts = parse_positive_sizes(args.seed_counts, label="seed-counts")
        if len(counts) > 8 or max(counts) > 32:
            raise ValueError("seed-counts may contain at most 8 sizes, each at most 32")
        if args.context >= args.corpus_tokens:
            raise ValueError("context must be shorter than corpus-tokens")
        policies = (
            ("replacement", "disjoint", "shifted")
            if args.sampling_policy == "all" else (args.sampling_policy,)
        )
        builders = {
            "replacement": prepare_evaluation_index_plan,
            "disjoint": prepare_disjoint_evaluation_index_plan,
            "shifted": prepare_shifted_evaluation_index_plan,
        }
        tokens = torch.arange(args.corpus_tokens, dtype=torch.long) % 7
        rows = []
        marginal_traces = []
        for policy in policies:
            for count in counts:
                plans = [
                    builders[policy](
                        tokens,
                        batch_size=args.batch_size,
                        block_size=args.context,
                        eval_batches=args.eval_batches,
                        seed=args.first_seed + i,
                    )
                    for i in range(count)
                ]
                coverage = analyze_cross_seed_overlap(plans)
                growth = analyze_coverage_growth(plans)
                marginal_traces.append({
                    "sampling_policy": policy,
                    "seed_count": count,
                    "steps": [asdict(step) for step in growth],
                })
                rows.append({
                    "sampling_policy": policy,
                    "seed_count": count,
                    "latest_seed_new_positions": growth[-1].new_corpus_token_positions,
                    "zero_new_coverage_seeds": sum(
                        step.new_corpus_token_positions == 0 for step in growth
                    ),
                    "global_unique_corpus_token_positions":
                        coverage.unique_corpus_token_positions,
                    "global_corpus_coverage_fraction": coverage.coverage_fraction,
                    "repeated_positions_within_seeds":
                        coverage.repeated_positions_within_seeds,
                    "repeated_positions_between_seeds":
                        coverage.repeated_positions_between_seeds,
                    "duplicate_starts_between_seeds":
                        coverage.duplicate_starts_between_seeds,
                    "total_window_token_positions":
                        coverage.total_window_token_positions,
                })
    except ValueError as exc:
        parser.error(str(exc))

    for row in rows:
        print(
            f"policy={row['sampling_policy']} seeds={row['seed_count']} "
            f"unique={row['global_unique_corpus_token_positions']} "
            f"coverage={row['global_corpus_coverage_fraction']:.3f} "
            f"between_seed_repeats={row['repeated_positions_between_seeds']}"
        )
    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", output)
    if args.json:
        report = build_benchmark_report(
            rows=rows, device=torch.device("cpu"),
            settings={
                "corpus_tokens": args.corpus_tokens,
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batches_per_seed": args.eval_batches,
                "first_seed": args.first_seed,
                "seed_counts": list(counts),
                "sampling_policy": args.sampling_policy,
                "contains_model_compute": False,
                "corpus_is_synthetic": True,
            },
        )
        report["marginal_coverage"] = marginal_traces
        report["limitations"] = (            "Sampler-only synthetic CPU experiment, no GPT forward or optimizer. "
            "Seeds reuse one underlying corpus and can share token positions, "
            "even under disjoint tiling. Global coverage does not measure "
            "statistical independence or support confidence intervals. "
            "Marginal coverage depends on the order of selected seeds."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
