"""Lesson 239: compare group rules with an explicitly supplied original GPT."""

import argparse

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizer_parity import compare_grouped_optimizer_layout
from rebuild_gpt.optimizers import configure_adamw
from rebuild_gpt.upstream_parity import load_upstream_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True,
                        help="Reviewed local original nanoGPT/model.py")
    args = parser.parse_args()
    cfg = GPTConfig(17, 8, 16, 4, 2, attention_impl="fused")
    ours = GPTLanguageModel(cfg)
    upstream = load_upstream_model(args.upstream_model, cfg)
    local_opt = configure_adamw(ours, learning_rate=0.001)
    reference_opt = upstream.configure_optimizers(
        weight_decay=0.1, learning_rate=0.001, betas=(0.9, 0.95),
        device_type="cpu",
    )
    print(compare_grouped_optimizer_layout(
        ours, upstream, local_opt, reference_opt, weight_decay=0.1,
    ))


if __name__ == "__main__":
    main()
