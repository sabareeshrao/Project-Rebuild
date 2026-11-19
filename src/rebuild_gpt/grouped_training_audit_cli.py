"""Compare several grouped-AdamW updates with a pinned original nanoGPT file."""

import argparse

from rebuild_gpt.grouped_training_audit import audit_grouped_training_trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True,
                        help="Reviewed original nanoGPT/model.py (executes Python code)")
    parser.add_argument("--upstream-sha256", required=True,
                        help="Exact SHA-256 of reviewed original code")
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--report", default="out/capstone/grouped_trajectory.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    args = parser.parse_args(argv)
    try:
        report = audit_grouped_training_trajectory(
            args.checkpoint, args.corpus, args.upstream_model,
            upstream_sha256=args.upstream_sha256, output_path=args.report,
            steps=args.steps, seed=args.seed, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        )
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        parser.error(str(exc))
    print("Grouped optimizer trajectory:", report["status"])
    print("Checked steps:", len(report["comparison"]["history"]))
    print("Audit:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
