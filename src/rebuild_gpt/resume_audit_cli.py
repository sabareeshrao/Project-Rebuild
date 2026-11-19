"""Train, resume and verify a tiny GPT in one CPU-only command."""

import argparse
from pathlib import Path

from rebuild_gpt.resume_audit import audit_resume_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt",
                        help="UTF-8 training corpus")
    parser.add_argument("--output-dir", default="out/resume_lab")
    parser.add_argument("--first-steps", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    try:
        audit = audit_resume_training(
            args.corpus, args.output_dir,
            first_steps=args.first_steps,
            total_steps=args.total_steps,
            seed=args.seed,
        )
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print("Interrupted-vs-uninterrupted training:", audit["status"])
    print("Compared tensor entries:", audit["checkpoint_comparison"]["matched_tensors"])
    print("Matched AdamW states:", audit["checkpoint_comparison"]["moment_parameters"])
    print("Evaluation records:", audit["evaluation_records"])
    print("Saved audit:", Path(args.output_dir) / "resume_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
