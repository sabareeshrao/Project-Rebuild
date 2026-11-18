"""Lesson 67: compare sampled tokens with a trusted upstream nanoGPT model.py."""

import argparse

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.generation_parity import compare_generation_against_upstream
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.upstream_parity import load_upstream_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True, help="Reviewed original nanoGPT model.py")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(17)
    config = GPTConfig(17, 5, 12, 3, 2, attention_impl="fused")
    ours = GPTLanguageModel(config)
    upstream = load_upstream_model(args.upstream_model, config)
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    parity = compare_generation_against_upstream(
        ours, upstream, prompt, 8,
        temperature=args.temperature, top_k=args.top_k, seed=33,
    )
    print("Matched tensors:", parity.matched_tensors)
    print("Generated next tokens:", parity.new_tokens)
    print("Temperature:", parity.temperature, "top-k:", parity.top_k)
    print("Exact sampled-token parity confirmed for this small CPU test.")


if __name__ == "__main__":
    main()
