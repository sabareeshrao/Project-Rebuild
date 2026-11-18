"""Lesson 92: demonstrate that split training matches uninterrupted training."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rebuild_gpt.text_training import train_char_gpt


def main() -> None:
    corpus = Path("data/tiny_story.txt")
    options = dict(
        block_size=4, n_embd=8, n_head=2, n_layer=1,
        batch_size=3, seed=11, learning_rate=0.02,
        eval_interval=2, eval_batches=2,
        gradient_accumulation_steps=2,
    )
    with TemporaryDirectory() as directory:
        state = Path(directory) / "training_state.pt"
        train_char_gpt(corpus, steps=2, training_state_path=state, **options)
        resumed, _, resumed_metrics = train_char_gpt(
            corpus, steps=4, resume_from=state, **options
        )
        complete, _, complete_metrics = train_char_gpt(
            corpus, steps=4, **options
        )
        equal = all(
            torch.equal(p, q) for p, q
            in zip(resumed.parameters(), complete.parameters())
        )
        print("Same model weights after resuming:", equal)
        print("Same validation metrics:", resumed_metrics == complete_metrics)
        assert equal
        assert resumed_metrics == complete_metrics


if __name__ == "__main__":
    main()
