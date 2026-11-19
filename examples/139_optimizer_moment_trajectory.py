"""Lesson 246: print AdamW momentum and variance drift after every step."""

import argparse

from rebuild_gpt.grouped_step_parity import compare_grouped_adamw_trajectory
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.training_parity_batches import training_parity_batches
from rebuild_gpt.upstream_parity import load_upstream_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    args = parser.parse_args()
    ours, tokenizer = load_text_checkpoint(args.checkpoint)
    batches = training_parity_batches(
        args.corpus, tokenizer, block_size=ours.config.block_size, steps=3,
    )
    upstream = load_upstream_model(args.upstream_model, ours.config)
    report = compare_grouped_adamw_trajectory(
        ours, upstream, *batches[0], steps=3, batches=batches,
    )
    for row in report["history"]:
        print(row)


if __name__ == "__main__":
    main()
