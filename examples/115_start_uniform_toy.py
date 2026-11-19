"""Compare raw balanced-sample and weighted scores to a known toy mean.

Synthetic scores vary by eligible-start region, not by token values.
For this deliberately simple population each region is constant,
so a region-weighted mean recovers the exact uniform-start score.
"""

import argparse
import csv
from pathlib import Path

import torch

from rebuild_gpt.balanced_evaluation import (
    prepare_balanced_evaluation_index_plan, start_region_bounds,
)
from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.start_uniform_estimator import estimate_start_uniform_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-tokens", type=int, default=25)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--region-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=228)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.corpus_tokens < 2 or args.context < 1
        or args.region_count < 1 or args.batch_size < 1
        or args.eval_batches < 1 or args.seed_count < 1 or args.first_seed < 0
    ):
        parser.error("positive corpus, context, region-count, batch-size, eval-batches and seed-count required; first-seed nonnegative")
    outs = [Path(p).resolve() for p in (args.csv, args.json) if p is not None]
    if len(set(outs)) != len(outs):
        parser.error("CSV and JSON output paths must differ")
    try:
        bounds = start_region_bounds(
            corpus_token_count=args.corpus_tokens,
            block_size=args.context, region_count=args.region_count,
        )
        tokens = torch.arange(args.corpus_tokens, dtype=torch.long)
        # Synthetic score is fixed for every start in each stratum.
        def score(start: int) -> float:
            return float(next(
                i + 1 for i, (lo, hi) in enumerate(bounds) if lo <= start < hi
            ))

        population = args.corpus_tokens - args.context
        exact = sum(score(start) for start in range(population)) / population
        rows = []
        for seed in range(args.first_seed, args.first_seed + args.seed_count):
            plan = prepare_balanced_evaluation_index_plan(
                tokens, batch_size=args.batch_size,
                block_size=args.context, eval_batches=args.eval_batches,
                region_count=args.region_count, seed=seed,
            )
            starts = [int(start) for batch in plan.starts for start in batch]
            estimates = estimate_start_uniform_score(
                starts, [score(start) for start in starts],
                corpus_token_count=args.corpus_tokens, block_size=args.context,
                region_count=args.region_count,
            )
            rows.append({
                "seed": seed,
                "sampled_windows": len(starts),
                "region_start_counts": ",".join(map(str, estimates.region_start_counts)),
                "exact_uniform_start_mean": exact,
                "raw_balanced_sample_mean": estimates.sampled_window_mean,
                "region_weighted_mean": estimates.start_uniform_estimate,
                "raw_absolute_error": abs(estimates.sampled_window_mean - exact),
                "weighted_absolute_error": abs(estimates.start_uniform_estimate - exact),
            })
    except ValueError as exc:
        parser.error(str(exc))
    for row in rows:
        print(
            f"seed={row['seed']} exact={exact:.6f} "
            f"raw={row['raw_balanced_sample_mean']:.6f} "
            f"weighted={row['region_weighted_mean']:.6f}"
        )
    if args.csv is not None:
        target = Path(args.csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", target)
    if args.json is not None:
        report = build_benchmark_report(
            rows=rows, device=torch.device("cpu"), settings={
                "corpus_token_count": args.corpus_tokens,
                "context": args.context,
                "region_count": args.region_count,
                "region_bounds": [list(bounds_item) for bounds_item in bounds],
                "batch_size": args.batch_size,
                "eval_batches": args.eval_batches,
                "first_seed": args.first_seed,
                "seed_count": args.seed_count,
                "synthetic_region_score": "region_index_plus_one",
                "exact_uniform_start_mean": exact,
                "contains_model_compute": False,
            },
        )
        report["limitations"] = (
            "Synthetic constant-within-region scores make the reweighted "
            "mean exact by construction. Real GPT losses vary within "
            "regions, so finite balanced samples will generally have "
            "sampling error. Overlapping windows and shared corpus content "
            "do not provide independent statistical trials or calibrated "
            "confidence intervals."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
