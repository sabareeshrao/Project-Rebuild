"""Lesson 235: opt-in comparison of one identical AdamW parameter update."""

import argparse

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.training_parity import compare_one_adamw_step
from rebuild_gpt.upstream_parity import load_upstream_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    ours, tokenizer = load_text_checkpoint(args.checkpoint)
    inputs, targets = prepare_validation_parity_batch(args.corpus, tokenizer, ours.config)
    upstream = load_upstream_model(args.upstream_model, ours.config)
    result = compare_one_adamw_step(
        ours, upstream, inputs, targets, learning_rate=args.learning_rate,
    )
    print("Maximum gradient error:", result.max_gradient_error)
    print("Maximum updated-weight error:", result.max_updated_weight_error)
    print("Largest local parameter update:", result.largest_local_parameter_change)


if __name__ == "__main__":
    main()
