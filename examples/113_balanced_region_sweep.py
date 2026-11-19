"""Measure how changing balanced-start region count affects token coverage.

Sampler-only CPU experiment on the last 20% of a UTF-8 corpus.
Start-position balance does not guarantee uniform token coverage.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from rebuild_gpt.balanced_evaluation import prepare_balanced_evaluation_index_plan
from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.sampling_policy_audit import audit_start_and_token_coverage
from rebuild_gpt.data import load_corpus, split_train_val
from rebuild_gpt.scaling_grid import parse_positive_sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--region-counts", default="2,4,8")
    parser.add_argument("--position-bins", type=int, default=8)
    parser.add_argument("--first-seed", type=int, default=222)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1 or args.context < 1 or args.eval_batches < 1
        or args.seed_count < 1 or args.first_seed < 0
    ):
        parser.error("batch-size, context, eval-batches and seed-count must be positive; first-seed nonnegative")
    source = Path(args.corpus).resolve()
    outputs = [
        Path(x).resolve() for x in (args.csv, args.json) if x is not None
    ]
    if len(set(outputs)) != len(outputs) or source in outputs:
        parser.error("output files must differ and cannot overwrite the corpus")
    try:
        counts = parse_positive_sizes(args.region_counts, label="region-counts")
        if len(counts) > 12 or max(counts) > 64:
            raise ValueError("region-counts supports at most 12 sizes up to 64")
        full_tokens, tokenizer = load_corpus(args.corpus)
        _, heldout = split_train_val(full_tokens, val_fraction=0.2)
        available_starts = heldout.numel() - args.context
        if available_starts < 1:
            raise ValueError("held-out corpus is too short for context")
        if not 1 <= args.position_bins <= min(256, heldout.numel()):
            raise ValueError("position-bins must be 1..min(256, held-out tokens)")
        requested = args.batch_size * args.eval_batches
        if max(counts) > requested:
            raise ValueError("region-counts cannot exceed the requested windows per seed")
        if max(counts) > available_starts:
            raise ValueError("region-counts cannot exceed eligible held-out start positions")

        rows = []
        detailed = []
        for regions in counts:
            plans = tuple(
                prepare_balanced_evaluation_index_plan(
                    heldout, batch_size=args.batch_size,
                    block_size=args.context, eval_batches=args.eval_batches,
                    seed=args.first_seed + index, region_count=regions,
                )
                for index in range(args.seed_count)
            )
            audit = audit_start_and_token_coverage(
                plans, start_regions=regions, token_bins=args.position_bins,
            )
            starts, overlap, positions = (
                audit.start_regions, audit.cross_seed_overlap, audit.token_positions,
            )
            rows.append({
                "sampling_regions": regions,
                "seed_count": args.seed_count,
                "windows_per_seed": requested,
                "aggregate_start_counts": ",".join(map(str, starts.aggregate_counts)),
                "seeds_with_empty_start_regions": starts.seeds_with_any_empty_region,
                "maximum_seed_start_imbalance": starts.maximum_seed_imbalance,
                "pooled_start_imbalance": starts.aggregate_imbalance,
                "global_unique_token_positions": overlap.unique_corpus_token_positions,
                "global_token_coverage_fraction": overlap.coverage_fraction,
                "repeated_positions_between_seeds": overlap.repeated_positions_between_seeds,
                "empty_token_coverage_bins": positions.zero_coverage_bins,
                "all_seeds_touch_every_start_region": (
                    audit.all_seeds_touch_every_start_region
                ),
                "start_balance_can_hide_token_gaps": (
                    audit.start_balance_without_full_token_bin_coverage
                ),
                "least_covered_token_bin_fraction": min(
                    bin.coverage_fraction for bin in positions.bins
                ),
            })
            detailed.append({
                "sampling_regions": regions,
                "starts_by_seed": [
                    {"seed": plan.seed, **asdict(summary)}
                    for plan, summary in zip(plans, starts.per_seed)
                ],
                "token_coverage_bins": [asdict(bin) for bin in positions.bins],
                "audit_flags": {
                    "all_seeds_touch_every_start_region":
                        audit.all_seeds_touch_every_start_region,
                    "any_unsampled_token_bin": audit.any_unsampled_token_bin,
                    "start_balance_can_hide_token_gaps":
                        audit.start_balance_without_full_token_bin_coverage,
                },
            })
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    for row in rows:
        print(
            f"regions={row['sampling_regions']} "
            f"max_seed_start_imbalance={row['maximum_seed_start_imbalance']} "
            f"global_token_coverage={row['global_token_coverage_fraction']:.3f} "
            f"empty_token_bins={row['empty_token_coverage_bins']}"
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
                "corpus": str(args.corpus),
                "vocabulary_size": tokenizer.vocab_size,
                "heldout_token_count": heldout.numel(),
                "validation_fraction": 0.2,
                "batch_size": args.batch_size,
                "context": args.context,
                "eval_batches": args.eval_batches,
                "region_counts": list(counts),
                "position_bins": args.position_bins,
                "first_seed": args.first_seed,
                "seed_count": args.seed_count,
                "sampling_policy": "balanced",
                "contains_model_compute": False,
            },
        )
        report["regional_details"] = detailed
        report["limitations"] = (
            "Start-region counts are explicitly balanced, but sampled windows "
            "can repeat or overlap. Start regions partition eligible start "
            "indices whereas position bins partition corpus tokens; they are "
            "different quantities: zero empty start regions can coexist with "
            "unsampled token bins. Samples reuse one held-out corpus across "
            "seeds. Coverage is descriptive, not a confidence interval, "
            "an independence test, or a claim of unbiased GPT validation."
        )
        write_benchmark_report(args.json, report)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
