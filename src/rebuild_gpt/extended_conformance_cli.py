"""Run eight independent GPT correctness checks against a saved checkpoint."""

import argparse

from rebuild_gpt.extended_conformance import run_extended_conformance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--report", default="out/conformance/extended_conformance.json")
    args = parser.parse_args(argv)
    try:
        result = run_extended_conformance(
            args.checkpoint, args.corpus,
            seed=args.seed, batch_size=args.batch_size,
            output_path=args.report,
        )
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print("Extended GPT conformance:", result["status"])
    print("Checks:", ", ".join(result["checks"]))
    print("Report:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
