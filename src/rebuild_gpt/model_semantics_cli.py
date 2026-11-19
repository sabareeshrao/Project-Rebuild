"""Audit four GPT semantics invariants on a trusted saved checkpoint."""

import argparse

from rebuild_gpt.model_semantics import run_model_semantics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/release_review/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="out/release_review/model_semantics.json")
    args = parser.parse_args(argv)
    try:
        report = run_model_semantics(
            args.checkpoint, args.corpus, seed=args.seed, output_path=args.output,
        )
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print("GPT model semantics:", report["status"])
    print("Checks:", ", ".join(report["checks"]))
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
