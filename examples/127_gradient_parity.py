"""Lesson 234: opt-in comparison against a reviewed original model.py."""

import argparse

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.training_parity import compare_gradients_against_upstream
from rebuild_gpt.upstream_parity import load_upstream_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    args = parser.parse_args()

    ours, tokenizer = load_text_checkpoint(args.checkpoint)
    inputs, targets = prepare_validation_parity_batch(
        args.corpus, tokenizer, ours.config,
    )
    upstream = load_upstream_model(args.upstream_model, ours.config)
    result = compare_gradients_against_upstream(ours, upstream, inputs, targets)
    print("Matched gradients:", result.matched_tensors)
    print("Maximum absolute error:", result.max_absolute_error)


if __name__ == "__main__":
    main()
