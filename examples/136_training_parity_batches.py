"""Lesson 243: inspect varying, reproducible optimizer-input batches."""

from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.training_parity_batches import training_parity_batches


def main() -> None:
    model, tokenizer = load_text_checkpoint("out/capstone/model.pt")
    batches = training_parity_batches(
        "data/tiny_story.txt", tokenizer,
        block_size=model.config.block_size, steps=3, seed=29,
    )
    for step, (inputs, targets) in enumerate(batches, start=1):
        print("step", step, "inputs", inputs.tolist())
        print("step", step, "targets", targets.tolist())


if __name__ == "__main__":
    main()
