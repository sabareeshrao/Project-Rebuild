"""Lesson 236: compare optimizer steps using SAVED GPT training weights."""

import argparse

from rebuild_gpt.saved_training_parity import compare_saved_capstone_adamw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True,
                        help="Locally reviewed original nanoGPT model.py")
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    report = compare_saved_capstone_adamw(
        args.checkpoint, args.corpus, args.upstream_model,
        learning_rate=args.learning_rate,
    )
    for name, value in report.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
