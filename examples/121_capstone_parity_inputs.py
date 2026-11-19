"""Lesson 228: reproduce the exact held-out tokens used for a parity test."""

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.text_checkpoint import load_text_checkpoint


def main() -> None:
    model, tokenizer = load_text_checkpoint("out/capstone/model.pt")
    x, y = prepare_validation_parity_batch(
        "data/tiny_story.txt", tokenizer, model.config, seed=29,
    )
    print("input shape:", tuple(x.shape))
    print("input IDs:", x.tolist())
    print("target IDs (-1 means ignored in loss):", y.tolist())


if __name__ == "__main__":
    main()
