"""Validate a saved GPT's extra numerical and long-context invariants."""

import argparse

from rebuild_gpt.post_milestone_audit import run_post_milestone_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/release_review/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--conformance", default="out/release_review/extended_conformance.json")
    parser.add_argument("--output", default="out/release_review/post_milestone.json")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--new-tokens", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        report = run_post_milestone_audit(
            args.checkpoint, args.corpus, args.conformance,
            output_path=args.output,
            seed=args.seed, new_tokens=args.new_tokens,
        )
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print("Post-milestone GPT audit:", report["status"])
    print("Gradient coordinates:", report["gradient_coordinates"]["checked_coordinates"])
    print("Numerical measurements:", report["numerical_budgets"]["checked_measurements"])
    print("Attention context:", report["long_prompt_generation"]["block_size"])
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
