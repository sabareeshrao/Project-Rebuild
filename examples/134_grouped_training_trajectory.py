"""Lesson 241: compare THREE updates using persistent AdamW optimizer state."""

import argparse

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.grouped_step_parity import compare_grouped_adamw_trajectory
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.upstream_parity import load_upstream_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    ours, tokenizer = load_text_checkpoint(args.checkpoint)
    x, y = prepare_validation_parity_batch(args.corpus, tokenizer, ours.config)
    upstream = load_upstream_model(args.upstream_model, ours.config)
    report = compare_grouped_adamw_trajectory(
        ours, upstream, x, y, steps=args.steps,
    )
    for row in report["history"]:
        print(row)


if __name__ == "__main__":
    main()
