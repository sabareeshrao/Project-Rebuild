"""Lesson 233: examine named gradients from one saved GPT training batch."""

import torch

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.training_parity import capture_upstream_named_gradients


def main() -> None:
    model, tokenizer = load_text_checkpoint("out/capstone/model.pt")
    inputs, targets = prepare_validation_parity_batch(
        "data/tiny_story.txt", tokenizer, model.config,
    )
    model.zero_grad(set_to_none=True)
    model.forward_nanogpt(inputs, targets)[1].backward()
    gradients = capture_upstream_named_gradients(model)
    print("unique gradient tensors:", len(gradients))
    for name, tensor in list(gradients.items())[:6]:
        print(name, "shape:", tuple(tensor.shape), "norm:", tensor.norm().item())


if __name__ == "__main__":
    main()
