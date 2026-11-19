"""Lesson 245: observe AdamW's first and second moments."""

import torch

from rebuild_gpt.adamw_state_parity import named_adamw_moments
from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.optimizers import configure_adamw
from rebuild_gpt.text_checkpoint import load_text_checkpoint


def main() -> None:
    model, tokenizer = load_text_checkpoint("out/capstone/model.pt")
    tokens, targets = prepare_validation_parity_batch(
        "data/tiny_story.txt", tokenizer, model.config,
    )
    optimizer = configure_adamw(model, learning_rate=0.001)
    model.forward_nanogpt(tokens, targets)[1].backward()
    optimizer.step()
    state = named_adamw_moments(model, optimizer)
    key = next(iter(state))
    print("parameter:", key)
    print("step counter:", state[key]["step"].item())
    print("first moment norm:", state[key]["exp_avg"].norm().item())
    print("second moment norm:", state[key]["exp_avg_sq"].norm().item())


if __name__ == "__main__":
    main()
