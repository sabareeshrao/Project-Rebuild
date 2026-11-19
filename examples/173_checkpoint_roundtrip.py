"""Lesson 280: verify a saved inference checkpoint survives a second save."""

import torch

from rebuild_gpt.checkpoint_roundtrip import check_checkpoint_roundtrip


def main() -> None:
    print(check_checkpoint_roundtrip(
        "out/capstone/model.pt", torch.tensor([[0, 1, 2]]),
    ))


if __name__ == "__main__":
    main()
