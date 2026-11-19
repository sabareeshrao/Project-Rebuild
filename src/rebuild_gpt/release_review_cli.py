"""Tiny-GPT release review: train once, then verify or reproduce evidence."""

import argparse
from pathlib import Path

from rebuild_gpt.capstone_evidence import run_review_capstone
from rebuild_gpt.capstone_spec import CapstoneSpec
from rebuild_gpt.review_evidence_verification import verify_review_evidence
from rebuild_gpt.review_reproduction import reproduce_review_evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "verify", "reproduce"),
                        default="verify")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--output-dir", default="out/release_review")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--training-seed", type=int, default=7)
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--conformance-seed", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        if args.mode == "train":
            spec = CapstoneSpec(
                steps=args.steps, seed=args.training_seed,
                prompt=args.prompt, new_tokens=args.new_tokens,
            )
            result = run_review_capstone(
                args.corpus, args.output_dir, spec=spec,
                conformance_seed=args.conformance_seed,
                batch_size=args.batch_size,
            )
            if result.get("status") != "passed" or result.get("checks_passed") != 8:
                raise AssertionError("Train-to-conformance review did not pass all checks")
            print("Trained and checked GPT:", result["status"])
            print("Model checkpoint:", result["checkpoint"])
        elif args.mode == "verify":
            result = verify_review_evidence(args.corpus, args.output_dir)
            if result.get("status") != "verified" or result.get("conformance_checks") != 8:
                raise AssertionError("Cross-report integrity check did not verify eight checks")
            print("Read-only review verification:", result["status"])
            print("Model SHA-256:", result["checkpoint_sha256"])
        else:
            result = reproduce_review_evidence(args.corpus, args.output_dir)
            if result.get("status") != "reproduced" or result.get("reproduced_checks") != 8:
                raise AssertionError("Independent reproduction did not verify eight checks")
            print("Independent conformance reproduction:", result["status"])
            print("Compared measurements:", result["compared_measurements"])
            print("Saved:", Path(args.output_dir) / "reproduction.json")
    except (ValueError, FileNotFoundError, FileExistsError, AssertionError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
