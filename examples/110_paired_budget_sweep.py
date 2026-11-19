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
    parser.add_argument(
        "--sampling-policy",
        choices=("replacement", "disjoint", "shifted", "balanced", "both", "all", "extended"),
        default="replacement",
        help="One policy, original pair (both), previous three (all), or four (extended)",
    )
    parser.add_argument(
        "--sampling-regions", type=int, default=None,
        help="Eligible-start region count when policy is balanced or extended",
    )
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
    if args.sampling_regions is not None:
        if args.sampling_policy not in ("balanced", "extended"):
            parser.error("--sampling-regions requires balanced or extended sampling-policy")
        if args.sampling_regions < 1:
            parser.error("--sampling-regions must be positive")
    region_count = args.sampling_regions if args.sampling_regions is not None else 4
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
        policies = {
            "both": ("replacement", "disjoint"),
            "all": ("replacement", "disjoint", "shifted"),
            "extended": ("replacement", "disjoint", "shifted", "balanced"),
        }.get(args.sampling_policy, (args.sampling_policy,))
        for policy in policies:
            for count in counts:
                result = compare_saved_checkpoints(
                    args.baseline, args.candidate, args.corpus,
                    seeds=chosen, batch_size=args.batch_size,
                    block_size=args.context, eval_batches=count,
                    device=args.device, sampling_policy=policy,
                    sampling_regions=region_count,
                )
                directions = result.directions.seed_mean_directions
                overlaps = result.paired.overlap_by_seed
                across = result.paired.cross_seed_overlap
                start_summary = result.paired.aggregate_start_regions
                rows.append({
                    "sampling_policy": policy,
                    "eval_batches_per_seed": count,
                    "sampling_regions": region_count if start_summary else None,
                    "start_region_counts_across_seeds": (
                        ",".join(map(str, start_summary.aggregate_counts))
                        if start_summary else None
                    ),
                    "maximum_seed_start_imbalance": (
                        start_summary.maximum_seed_imbalance
                        if start_summary else None
                    ),
                    "seeds_with_empty_start_regions": (
                        start_summary.seeds_with_any_empty_region
                        if start_summary else None
                    ),
                    "lattice_phases_by_seed": ",".join(
                        str(lattice.phase) if lattice is not None else "n/a"
                        for lattice in result.paired.lattice_by_seed
                    ),
                    "feasible_phase_counts_by_seed": ",".join(
                        str(lattice.phase_count_feasible_for_budget)
                        if lattice is not None else "n/a"
                        for lattice in result.paired.lattice_by_seed
                    ),
                    "seeds": args.seed_count,
                    "global_unique_corpus_token_positions": across.unique_corpus_token_positions,
                    "global_corpus_coverage_fraction": across.coverage_fraction,
                    "repeated_positions_between_seeds": across.repeated_positions_between_seeds,
                    "duplicate_starts_between_seeds": across.duplicate_starts_between_seeds,
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
                "sampling_regions": (
                    region_count if "balanced" in policies else None
                ),
                "first_seed": args.first_seed,
                "seed_count": args.seed_count,
                "validation_fraction": 0.2,
                "optimizer_updates_during_comparison": 0,
                "sampled_windows_share_seeded_prefixes": (
                    args.sampling_policy not in ("shifted", "all", "balanced", "extended")
                ),
                "shifted_phase_set_can_change_with_budget": (
                    args.sampling_policy in ("shifted", "all", "extended")
                ),
                "balanced_region_allocation_can_change_with_budget": (
                    args.sampling_policy in ("balanced", "extended")
                ),
            },
        )
        report["limitations"] = (
            "Replacement and fixed-origin disjoint plans reuse seeded "
            "sampling prefixes as the budget increases, so their rows "
            "are dependent. Shifted lattice phases depend on the requested "
            "count: different budgets need not share prefixes. The three "
            "policies sample different distributions; shifted and fixed "
            "policies use T+1 tiles. Balanced sampling equalizes start "
            "counts across regions but permits overlapping windows and "
            "changing budget allocations, so its budgets also need not "
            "share prefixes. All observations still come from one "
            "held-out corpus. The legacy total_unique_corpus_positions_"
            "across_seed_plans column is NOT a global union: it sums "
            "within-seed unique coverage. Use global_unique_corpus_token_positions "
            "for the deduplicated union and repeated_positions_between_seeds "
            "for cross-seed reuse. These are descriptive estimates, "
            "not calibrated confidence intervals."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    print(
        "Replacement/fixed-disjoint budgets reuse the same seeded "
        "sampling prefix; shifted phases and balanced allocations "
        "can change with budget."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
