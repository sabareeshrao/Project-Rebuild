"""Lesson 231: create an audit only if two genuine parity checks succeed."""

import argparse
import json

from rebuild_gpt.capstone_parity_audit import run_capstone_parity_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--upstream-sha256", default=None)
    parser.add_argument("--report", default="out/capstone/upstream_parity.json")
    args = parser.parse_args()
    report = run_capstone_parity_audit(
        args.checkpoint, args.corpus, args.upstream_model,
        expected_upstream_sha256=args.upstream_sha256,
        output_path=args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
