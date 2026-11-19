"""Review GPT conformance evidence without confusing verification and training."""

import argparse
from pathlib import Path

from rebuild_gpt.conformance_integrity import verify_conformance_integrity
from rebuild_gpt.conformance_reproduction import reproduce_conformance_report
from rebuild_gpt.conformance_seed_sweep import run_conformance_seed_sweep


def parse_seeds(value: str) -> tuple[int, ...]:
    """Parse distinct comma-separated nonnegative decimal seeds."""
    parts = [token.strip() for token in value.split(",")]
    if (len(parts) < 2 or any(not p or not p.isdecimal() for p in parts)):
        raise ValueError("Seeds must be at least two comma-separated nonnegative integers")
    seeds = tuple(int(part) for part in parts)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed values must be distinct")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "reproduce", "sweep"),
                        default="verify")
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--report", default="out/conformance/extended_conformance.json")
    parser.add_argument("--seed-sweep", default="11,29")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="out/conformance/seed_sweep.json")
    args = parser.parse_args(argv)
    try:
        if args.mode == "verify":
            result = verify_conformance_integrity(
                args.report, args.checkpoint, args.corpus,
            )
            print("Read-only artifact verification:", result["status"])
            print("Checked eight-check report:", result["check_count"])
        elif args.mode == "reproduce":
            result = reproduce_conformance_report(
                args.report, args.checkpoint, args.corpus,
            )
            print("Independent conformance reproduction:", result["status"])
            print("Checks reproduced:", result["reproduced_checks"])
        else:
            seeds = parse_seeds(args.seed_sweep)
            result = run_conformance_seed_sweep(
                args.checkpoint, args.corpus,
                seeds=seeds, batch_size=args.batch_size,
                output_path=args.output,
            )
            print("Multiple-seed conformance:", result["status"])
            print("Verified seeds:", result["seeds"])
            print("Report:", Path(args.output))
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
