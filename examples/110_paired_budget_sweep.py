"""Compare two saved GPTs as sampled validation batch count increases.

All budgets reuse the same prefix of random draws for each seed.
These are dependent comparisons, NOT independent significance tests.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.checkpoint_comparison import compare_saved_checkpoints
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.scaling_grid import parse_positive_sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Self-contained baseline model .pt")
    parser.add_argument("--candidate", required=True, help="Self-contained candidate model .pt")
    parser.add_argument("--corpus", required=True, help="UTF-8 text corpus")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--eval-batch-counts", default="1,2,4")
    parser.add_argument("--sampling-policy", choices=("replacement", "disjoint", "both"),
                        default="replacement", help="Compare original, disjoint, or both policies")
    parser.add_argument("--first-seed", type=int, default=187)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1 or args.context < 1 or args.seed_count < 1
        or args.first_seed < 0
    ):
        parser.error("batch-size, context and seed-count must be positive; first-seed nonnegative")
    inputs = {Path(x).resolve() for x in (args.baseline, args.candidate, args.corpus)}
    outputs = [Path(x).resolve() for x in (args.csv, args.json) if x is not None]
    if len(set(outputs)) != len(outputs) or any(p in inputs for p in outputs):
        parser.error("CSV and JSON paths must differ and cannot overwrite inputs")
    try:
        counts = parse_positive_sizes(args.eval_batch_counts, label="eval-batch-counts")
        if len(counts) > 8 or max(counts) > 64:
            raise ValueError("eval-batch-counts is limited to eight sizes of at most 64")
        device = resolve_device(args.device)
        chosen = tuple(range(args.first_seed, args.first_seed + args.seed_count))
        rows = []
        policies = (
            ("replacement", "disjoint")
            if args.sampling_policy == "both" else (args.sampling_policy,)
        )
        for policy in policies:
            for count in counts:
                result = compare_saved_checkpoints(
                    args.baseline, args.candidate, args.corpus,
                    seeds=chosen, batch_size=args.batch_size,
                    block_size=args.context, eval_batches=count,
                    device=args.device, sampling_policy=policy,
                )
                directions = result.directions.seed_mean_directions
                overlaps = result.paired.overlap_by_seed
                rows.append({
                    "sampling_policy": policy,
                    "eval_batches_per_seed": count,
                    "seeds": args.seed_count,
                    "sampled_target_positions_per_model":
                        count * args.seed_count * args.batch_size * args.context,
                    "mean_candidate_minus_baseline": result.paired.per_seed_summary.mean,
                    "sample_stdev_across_seed_means":
                        result.paired.per_seed_summary.sample_stdev,
                    "improved_seed_means": directions.improved,
                    "worsened_seed_means": directions.worsened,
                    "tied_seed_means": directions.tied,
                    "duplicate_starts_across_seeds": sum(
                        item.duplicate_start_count for item in overlaps
                    ),
                    "repeated_window_positions_across_seeds": sum(
                        item.repeated_window_token_positions for item in overlaps
                    ),
                    "total_unique_corpus_positions_across_seed_plans": sum(
                        item.unique_corpus_token_positions for item in overlaps
                    ),
                })
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    for row in rows:
        print(
            f"policy={row['sampling_policy']} "
            f"eval_batches={row['eval_batches_per_seed']} "
            f"paired_mean_delta={row['mean_candidate_minus_baseline']:+.6f} "
            f"seed_stdev={row['sample_stdev_across_seed_means']}"
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
            rows=rows, device=device, settings={
                "baseline_checkpoint": str(args.baseline),
                "candidate_checkpoint": str(args.candidate),
                "corpus": str(args.corpus),
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batch_counts": list(counts),
                "sampling_policy": args.sampling_policy,
                "first_seed": args.first_seed,
                "seed_count": args.seed_count,
                "validation_fraction": 0.2,
                "optimizer_updates_during_comparison": 0,
                "sampled_windows_share_seeded_prefixes": True,
            },
        )
        report["limitations"] = (
            "Larger budgets reuse earlier sampled windows for each seed, "
            "so sweep rows are dependent. Replacement and disjoint sample "
            "different distributions; disjoint windows use a fixed T+1 "
            "grid. All windows still come from one held-out corpus, and "
            "different seeds may reuse tiles. Overlap totals are summed "
            "within seeds, not globally de-duplicated. Differences are "
            "descriptive, not calibrated confidence intervals."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print("Validation-batch counts reuse the same seeded sampling prefix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
