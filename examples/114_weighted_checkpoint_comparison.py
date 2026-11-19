"""Compare two saved GPT models using population-weighted balanced samples.

The raw and reweighted differences are descriptive paired estimates.
Token windows can overlap; no calibrated confidence interval is produced.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from rebuild_gpt.balanced_evaluation import prepare_balanced_evaluation_index_plan
from rebuild_gpt.benchmark_report import build_benchmark_report, write_benchmark_report
from rebuild_gpt.data import split_train_val
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.stratified_model_comparison import compare_balanced_window_losses
from rebuild_gpt.text_checkpoint import load_text_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--sampling-regions", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=227)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.context < 1 or args.batch_size < 1 or args.eval_batches < 1
        or args.sampling_regions < 1 or args.seed_count < 1
        or args.first_seed < 0
    ):
        parser.error("context, batch-size, eval-batches, sampling-regions and seed-count must be positive; first-seed nonnegative")
    inputs = {Path(s).resolve() for s in (args.baseline, args.candidate, args.corpus)}
    outputs = [Path(s).resolve() for s in (args.csv, args.json) if s is not None]
    if len(set(outputs)) != len(outputs) or any(s in inputs for s in outputs):
        parser.error("output paths must differ and cannot overwrite checkpoints or corpus")
    try:
        device = resolve_device(args.device)
        with torch.random.fork_rng(devices=[]):
            baseline, original_vocab = load_text_checkpoint(args.baseline)
            candidate, candidate_vocab = load_text_checkpoint(args.candidate)
        if original_vocab.vocab != candidate_vocab.vocab:
            raise ValueError("checkpoint vocabularies must match exactly (character order)")
        text = Path(args.corpus).read_text(encoding="utf-8")
        tokens = torch.tensor(original_vocab.encode(text), dtype=torch.long)
        _, validation = split_train_val(tokens, val_fraction=0.2)
        if validation.numel() <= args.context:
            raise ValueError("held-out split must exceed the requested context")
        if args.context > min(baseline.config.block_size, candidate.config.block_size):
            raise ValueError("context exceeds the model block_size")
        rows = []
        regions = []
        baseline, candidate = baseline.to(device), candidate.to(device)
        for seed in range(args.first_seed, args.first_seed + args.seed_count):
            plan = prepare_balanced_evaluation_index_plan(
                validation,
                batch_size=args.batch_size, block_size=args.context,
                eval_batches=args.eval_batches, region_count=args.sampling_regions,
                seed=seed,
            )
            report = compare_balanced_window_losses(
                baseline, candidate, plan,
                region_count=args.sampling_regions,
            )
            delta = report.candidate_minus_baseline
            rows.append({
                "seed": seed,
                "sampling_regions": args.sampling_regions,
                "sampled_windows": delta.sampled_window_count,
                "raw_candidate_minus_baseline": report.raw_paired_delta,
                "weighted_candidate_minus_baseline": report.corrected_paired_delta,
                "reweighting_change": delta.reweighting_difference,
                "raw_baseline_loss": report.baseline.sampled_window_mean,
                "weighted_baseline_loss": report.baseline.start_uniform_estimate,
                "raw_candidate_loss": report.candidate.sampled_window_mean,
                "weighted_candidate_loss": report.candidate.start_uniform_estimate,
            })
            regions.append({
                "seed": seed,
                "region_start_counts": delta.region_start_counts,
                "region_population_sizes": delta.region_population_sizes,
                "region_population_weights": delta.region_population_weights,
                "region_mean_paired_deltas": delta.region_mean_scores,
            })
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    for row in rows:
        print(
            f"seed={row['seed']} raw_delta={row['raw_candidate_minus_baseline']:+.6f} "
            f"weighted_delta={row['weighted_candidate_minus_baseline']:+.6f}"
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
        payload = build_benchmark_report(
            rows=rows, device=device, settings={
                "baseline": str(args.baseline),
                "candidate": str(args.candidate),
                "corpus": str(args.corpus),
                "vocabulary_size": original_vocab.vocab_size,
                "validation_fraction": 0.2,
                "validation_token_count": validation.numel(),
                "context": args.context,
                "batch_size": args.batch_size,
                "eval_batches": args.eval_batches,
                "first_seed": args.first_seed,
                "seed_count": args.seed_count,
                "sampling_regions": args.sampling_regions,
                "sampling_policy": "balanced",
                "window_score": "mean_next_token_cross_entropy_per_window",
                "optimizer_updates_during_comparison": 0,
            },
        )
        payload["region_diagnostics"] = regions
        payload["limitations"] = (
            "Reweighting by eligible-start population estimates a uniform-start "
            "mean only when draws are uniform within regions and window scores "
            "are deterministic. Repeated and overlapping windows from one "
            "held-out corpus are not independent observations. These "
            "descriptive paired estimates are not confidence intervals or "
            "generalization guarantees."
        )
        write_benchmark_report(args.json, payload)
        print("JSON:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
