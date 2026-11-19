"""One command to prove a GPT training run survives multiple restarts."""

import argparse
from pathlib import Path

from rebuild_gpt.multistage_resume_audit import audit_multistage_resume


def parse_stops(text: str) -> tuple[int, ...]:
    """Parse comma-separated interruption steps; reject empty segments."""
    try:
        fields = [part.strip() for part in text.split(",")]
        if not fields or any(not field or not field.isdecimal() for field in fields):
            raise ValueError
        return tuple(int(field) for field in fields)
    except (AttributeError, ValueError) as exc:
        raise ValueError("stops must be comma-separated positive integers") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--output-dir", default="out/multistage_resume")
    parser.add_argument("--total-steps", type=int, default=6)
    parser.add_argument("--stops", default="2,4", help="e.g. 2,4")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    try:
        report = audit_multistage_resume(
            args.corpus, args.output_dir,
            total_steps=args.total_steps,
            stops=parse_stops(args.stops),
            seed=args.seed,
        )
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print("Multi-interruption GPT training:", report["status"])
    print("Verified stages:", [stage["step"] for stage in report["stages"]])
    print("Matching saved state:", report["checkpoint_comparison"]["identical"])
    print("Audit:", Path(args.output_dir) / "multistage_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
