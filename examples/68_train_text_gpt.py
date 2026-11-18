"""Lesson 71: train a character-level GPT and save its inference checkpoint."""

import argparse

from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.text_training import train_char_gpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/tiny_story.txt")
    parser.add_argument("--output", default="tiny_story_gpt.pt")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    model, tokenizer, metrics = train_char_gpt(args.corpus, steps=args.steps)
    save_text_checkpoint(model, tokenizer, args.output)
    print("character vocabulary size:", tokenizer.vocab_size)
    print("validation loss before:", round(metrics["validation_before"], 4))
    print("validation loss after: ", round(metrics["validation_after"], 4))
    print("saved inference bundle:", args.output)


if __name__ == "__main__":
    main()
