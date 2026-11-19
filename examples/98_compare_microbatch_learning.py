"""Compare real training trajectories across logical microbatch sizes.

Every experiment uses the same corpus, logical batch, seed and model.
Small floating-point differences are reported rather than hidden.
"""

import argparse
import json
from pathlib import Path

import torch

from rebuild_gpt.scaling_grid import parse_positive_sizes
from rebuild_gpt.text_training import train_char_gpt


def run_experiment(
    corpus: str | Path, *, steps: int, batch_size: int, block_size: int,
    sizes: tuple[int, ...], seed: int, eval_every: int, device: str,
) -> dict:
    """Train a baseline and independent streamed variants from identical seeds."""
    config = dict(
        steps=steps, block_size=block_size, batch_size=batch_size,
        n_embd=8, n_head=2, n_layer=1,
        seed=seed, eval_interval=eval_every, eval_batches=2,
        attention_impl="sdpa", device=device, learning_rate=0.001,
    )
    baseline_history = []
    original, tokenizer, baseline_metrics = train_char_gpt(
        corpus, history=baseline_history, **config,
    )
    experiments = []
    for size in sizes:
        trajectory = []
        candidate, _, metrics = train_char_gpt(
            corpus, history=trajectory, microbatch_size=size, **config,
        )
        differences = [
            (p.detach() - q.detach()).abs().max().item()
            for p, q in zip(original.parameters(), candidate.parameters())
        ]
        baseline_validation = [entry["validation_loss"] for entry in baseline_history]
        current_validation = [entry["validation_loss"] for entry in trajectory]
        assert [entry["step"] for entry in trajectory] == [
            entry["step"] for entry in baseline_history
        ]
        experiments.append({
            "microbatch_size": size,
            "chunks_per_step": (batch_size + size - 1) // size,
            "metrics": metrics,
            "validation_history": trajectory,
            "max_parameter_absolute_difference": max(differences, default=0.0),
            "max_validation_loss_absolute_difference": max(
                (abs(a - b) for a, b in zip(baseline_validation, current_validation)),
                default=0.0,
            ),
        })
    return {
        "schema_version": 1,
        "settings": {
            "corpus": str(corpus), "steps": steps, "batch_size": batch_size,
            "block_size": block_size, "microbatch_sizes": list(sizes),
            "seed": seed, "eval_every": eval_every,
            "device": device, "attention_impl": "sdpa",
            "model_dimensions": {"n_embd": 8, "n_head": 2, "n_layer": 1},
        },
        "vocab_size": tokenizer.vocab_size,
        "baseline": {
            "metrics": baseline_metrics,
            "validation_history": baseline_history,
        },
        "microbatch_results": experiments,
        "note": (
            "Results use identical corpus, logical batch and seed. Floating-point "
            "rounding may differ by chunk size or device; no speedup is implied."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--microbatch-sizes", default="1,2,5")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=145)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if (
        args.steps < 1 or args.batch_size < 1 or args.block_size < 1
        or args.eval_every < 1 or args.seed < 0
    ):
        parser.error("steps, batch-size, block-size and eval-every must be positive; seed nonnegative")
    try:
        sizes = parse_positive_sizes(args.microbatch_sizes, label="microbatch-sizes")
        if any(size > args.batch_size for size in sizes):
            raise ValueError("microbatch sizes must not exceed batch-size")
        result = run_experiment(
            args.corpus, steps=args.steps, batch_size=args.batch_size,
            block_size=args.block_size, sizes=sizes, seed=args.seed,
            eval_every=args.eval_every, device=args.device,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    for row in result["microbatch_results"]:
        print(
            f"microbatch={row['microbatch_size']} chunks={row['chunks_per_step']} "
            f"validation={row['metrics']['validation_after']:.6f} "
            f"max_weight_diff={row['max_parameter_absolute_difference']:.3g} "
            f"max_val_gap={row['max_validation_loss_absolute_difference']:.3g}"
        )
    if args.json is not None:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("JSON:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
