"""Evaluate two saved GPT inference bundles on matched validation samples.

The report uses the exact same held-out token IDs for both models.
Descriptive paired deltas are NOT statistical significance tests.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.checkpoint_comparison import compare_saved_checkpoints
from rebuild_gpt.devices import resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Saved baseline inference .pt")
    parser.add_argument("--candidate", required=True, help="Saved candidate inference .pt")
    parser.add_argument("--corpus", required=True, help="UTF-8 text for held-out evaluation")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=186)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument(
        "--sampling-policy",
        choices=("replacement", "disjoint", "shifted", "balanced"),
        default="replacement",
        help="Original random starts, fixed/shifted tiles, or balanced regional starts",
    )
    parser.add_argument(
        "--sampling-regions", type=int, default=None,
        help="Positive number of eligible-start regions; only for balanced sampling",
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1 or args.context < 1 or args.eval_batches < 1
        or args.seed_count < 1 or args.first_seed < 0
    ):
        parser.error("batch-size, context, eval-batches and seed-count must be positive; first-seed nonnegative")
    if args.sampling_regions is not None:
        if args.sampling_policy != "balanced":
            parser.error("--sampling-regions requires --sampling-policy balanced")
        if args.sampling_regions < 1:
            parser.error("--sampling-regions must be positive")
    sources = {Path(p).resolve() for p in (args.baseline, args.candidate, args.corpus)}
    outputs = [Path(p).resolve() for p in (args.csv, args.json) if p is not None]
    if len(set(outputs)) != len(outputs) or any(p in sources for p in outputs):
        parser.error("CSV and JSON outputs must be distinct and cannot overwrite inputs")
    try:
        device = resolve_device(args.device)
        result = compare_saved_checkpoints(
            args.baseline, args.candidate, args.corpus,
            seeds=tuple(range(args.first_seed, args.first_seed + args.seed_count)),
            batch_size=args.batch_size, block_size=args.context,
            eval_batches=args.eval_batches, device=args.device,
            sampling_policy=args.sampling_policy,
            sampling_regions=args.sampling_regions if args.sampling_regions is not None else 4,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    rows = []
    for seed, baseline_loss, candidate_loss, delta, batch_deltas, overlap, lattice, starts in zip(
        result.paired.seeds, result.paired.baseline_losses,
        result.paired.candidate_losses,
        result.paired.per_seed_means, result.paired.differences,
        result.paired.overlap_by_seed, result.paired.lattice_by_seed,
        result.paired.start_regions_by_seed,
    ):
        rows.append({
            "seed": seed,
            "sampling_policy": args.sampling_policy,
            "sampling_regions": result.sampling_regions,
            "start_region_counts": (
                ",".join(map(str, starts.counts)) if starts is not None else None
            ),
            "empty_start_regions": starts.empty_regions if starts is not None else None,
            "start_region_imbalance": starts.imbalance if starts is not None else None,
            "lattice_phase": lattice.phase if lattice is not None else None,
            "feasible_lattice_phases": (
                lattice.phase_count_feasible_for_budget
                if lattice is not None else None
            ),
            "candidate_tiles_at_phase": (
                lattice.candidate_tiles_at_phase if lattice is not None else None
            ),
            "duplicate_starts": overlap.duplicate_start_count,
            "repeated_window_token_positions": overlap.repeated_window_token_positions,
            "unique_corpus_token_positions": overlap.unique_corpus_token_positions,
            "baseline_mean_loss": sum(baseline_loss) / len(baseline_loss),
            "candidate_mean_loss": sum(candidate_loss) / len(candidate_loss),
            "candidate_minus_baseline": delta,
            "improved_batches": sum(x < 0 for x in batch_deltas),
            "worsened_batches": sum(x > 0 for x in batch_deltas),
            "tied_batches": sum(x == 0 for x in batch_deltas),
        })
    for row in rows:
        print(
            f"seed={row['seed']} baseline={row['baseline_mean_loss']:.6f} "
            f"candidate={row['candidate_mean_loss']:.6f} "
            f"delta={row['candidate_minus_baseline']:+.6f}"
        )
    print(
        f"paired mean delta={result.paired.per_seed_summary.mean:+.6f}; "
        f"improved batches={result.directions.batch_directions.improved}/"
        f"{result.directions.batch_directions.total}"
    )
    global_coverage = result.paired.cross_seed_overlap
    print(
        f"global unique token positions={global_coverage.unique_corpus_token_positions}; "
        f"repeated between seeds={global_coverage.repeated_positions_between_seeds}"
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
                "vocabulary_size": result.vocabulary_size,
                "validation_token_count": result.validation_token_count,
                "baseline_config": result.baseline_config,
                "candidate_config": result.candidate_config,
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batches_per_seed": args.eval_batches,
                "first_seed": args.first_seed,
                "seed_count": args.seed_count,
                "sampling_policy": args.sampling_policy,
                "sampling_regions": result.sampling_regions,
                "validation_fraction": 0.2,
                "optimizer_updates_during_comparison": 0,
            },
        )
        report["paired_summary"] = asdict(result.paired.per_seed_summary)
        report["window_overlap_by_seed"] = [
            {"seed": seed, **asdict(overlap)}
            for seed, overlap in zip(result.paired.seeds, result.paired.overlap_by_seed)
        ]
        report["cross_seed_overlap"] = {
            **asdict(global_coverage),
            "coverage_fraction": global_coverage.coverage_fraction,
            "total_repeated_positions": global_coverage.total_repeated_positions,
        }
        report["start_regions_by_seed"] = [
            {"seed": seed, **asdict(starts)} if starts is not None
            else {"seed": seed, "regions": None}
            for seed, starts in zip(
                result.paired.seeds, result.paired.start_regions_by_seed,
            )
        ]
        report["lattice_by_seed"] = [
            {"seed": seed, **asdict(lattice)} if lattice is not None
            else {"seed": seed, "lattice": None}
            for seed, lattice in zip(
                result.paired.seeds, result.paired.lattice_by_seed,
            )
        ]
        directions = asdict(result.directions)
        # Dataclass properties (total/improved_fraction) are not
        # included by asdict; expose them explicitly for report readers.
        for group in ("batch_directions", "seed_mean_directions"):
            counts = directions[group]
            counts["total"] = sum(counts[key] for key in ("improved", "worsened", "tied"))
            counts["improved_fraction"] = counts["improved"] / counts["total"]
        report["direction_diagnostics"] = directions
        report["limitations"] = (
            "With-replacement windows can overlap. Fixed and shifted "
            "T+1 tile lattices change the sampling distribution. Balanced "
            "sampling allocates nearly equal start counts to corpus regions "
            "but can still overlap and does not ensure equal token coverage. "
            "Shifted "
            "lattice feasible phases depend on the requested sample count. "
            "All samples still come from one held-out corpus. Per-seed "
            "overlap counts do not capture shared coverage between seeds; "
            "global repeated positions include both. Paired differences "
            "and improvement counts are not independent observations "
            "or statistical significance tests."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
