"""Check a saved GPT's causal semantics, batch isolation and backprop."""

import argparse

from rebuild_gpt.model_conformance import run_model_conformance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--report", default="out/conformance/model_conformance.json")
    args = parser.parse_args(argv)
    try:
        report = run_model_conformance(
            args.checkpoint, args.corpus,
            batch_size=args.batch_size, seed=args.seed,
            output_path=args.report,
        )
    except (ValueError, FileNotFoundError, AssertionError) as exc:
        parser.error(str(exc))
    print("Model conformance:", report["status"])
    for name, result in report["checks"].items():
        print(f"{name}: {result}")
    print("Audit:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
