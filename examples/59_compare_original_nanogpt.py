"""Lesson 62: run our GPT against an explicitly supplied original nanoGPT model.py."""

import argparse
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.upstream_parity import compare_against_upstream, load_upstream_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True, help="Path to a trusted original nanoGPT model.py")
    parser.add_argument("--bias-free", action="store_true", help="Compare the optional bias=False variant")
    args = parser.parse_args()
    torch.manual_seed(17)
    config = GPTConfig(
        vocab_size=17,
        block_size=6,
        n_embd=12,
        n_head=3,
        n_layer=2,
        dropout=0.1,
        bias=not args.bias_free,
        attention_impl="fused",
    )
    ours = GPTLanguageModel(config)
    reference = load_upstream_model(args.upstream_model, config)
    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    targets = torch.tensor([[1, 2, 3, 4, -1]])
    report = compare_against_upstream(ours, reference, tokens, targets)
    print("Matched tensors:", report.matched_tensors)
    print("Maximum absolute logit difference:", report.max_logit_error)
    print("Absolute cross-entropy difference:", report.loss_error)
    print("Numerical parity confirmed for this small CPU configuration.")


if __name__ == "__main__":
    main()
