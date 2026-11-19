"""Lesson 229: compare saved training logits and masked loss against nanoGPT."""

import argparse

from rebuild_gpt.capstone_parity import compare_capstone_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True,
                        help="Path to a reviewed, trusted original nanoGPT model.py")
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    args = parser.parse_args()
    report = compare_capstone_forward(args.checkpoint, args.corpus, args.upstream_model)
    for name, value in report.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
