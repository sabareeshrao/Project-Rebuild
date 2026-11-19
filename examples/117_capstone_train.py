"""Lesson 224: run a few actual GPT optimizer steps on the tiny corpus."""

from rebuild_gpt.capstone import train_capstone
from rebuild_gpt.capstone_spec import CapstoneSpec


def main() -> None:
    model, tokenizer, metrics = train_capstone(
        "data/tiny_story.txt", CapstoneSpec(steps=3)
    )
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    print("vocabulary size:", tokenizer.vocab_size)
    print("held-out losses:", metrics["validation_before"], metrics["validation_after"])


if __name__ == "__main__":
    main()
