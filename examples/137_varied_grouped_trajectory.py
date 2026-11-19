"""Lesson 244: compare multiple REAL training batches against reviewed nanoGPT."""

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
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    ours, tokenizer = load_text_checkpoint(args.checkpoint)
    batches = training_parity_batches(
        args.corpus, tokenizer, steps=args.steps,
        block_size=ours.config.block_size,
    )
    reference = load_upstream_model(args.upstream_model, ours.config)
    result = compare_grouped_adamw_trajectory(
        ours, reference, *batches[0], steps=len(batches), batches=batches,
    )
    for item in result["history"]:
        print(item)


if __name__ == "__main__":
    main()
