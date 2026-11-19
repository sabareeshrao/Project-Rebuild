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
    parser.add_argument("--sampling-policy", choices=("replacement", "disjoint"),
                        default="replacement", help="Original random starts or nonoverlapping T+1 tiles")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1 or args.context < 1 or args.eval_batches < 1
        or args.seed_count < 1 or args.first_seed < 0
    ):
        parser.error("batch-size, context, eval-batches and seed-count must be positive; first-seed nonnegative")
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
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    rows = []
    for seed, baseline_loss, candidate_loss, delta, batch_deltas, overlap in zip(
        result.paired.seeds, result.paired.baseline_losses,
        result.paired.candidate_losses,
        result.paired.per_seed_means, result.paired.differences,
        result.paired.overlap_by_seed,
    ):
        rows.append({
            "seed": seed,
            "sampling_policy": args.sampling_policy,
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
                "validation_fraction": 0.2,
                "optimizer_updates_during_comparison": 0,
            },
        )
        report["paired_summary"] = asdict(result.paired.per_seed_summary)
        report["window_overlap_by_seed"] = [
            {"seed": seed, **asdict(overlap)}
            for seed, overlap in zip(result.paired.seeds, result.paired.overlap_by_seed)
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
            "With-replacement windows can overlap; disjoint windows use "
            "a fixed T+1 tiling that changes the sampling distribution. "
            "All samples still come from one held-out corpus. Paired "
            "differences and improvement counts are descriptive, not "
            "independent trials or statistical significance tests."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
