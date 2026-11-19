"""One saved-GPT optimizer parity check against a reviewed nanoGPT model.py."""

import argparse

from rebuild_gpt.training_parity_audit import run_training_step_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--upstream-sha256", default=None,
                        help="Optional SHA-256 pin checked before importing Python")
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--report", default="out/capstone/adamw_parity.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        report = run_training_step_audit(
            args.checkpoint, args.corpus, args.upstream_model,
            expected_upstream_sha256=args.upstream_sha256,
            output_path=args.report,
            batch_size=args.batch_size, seed=args.seed,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        )
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        parser.error(str(exc))
    print("Training-step parity:", report["status"])
    print("Comparisons:", report["comparison"])
    print("Audit:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
