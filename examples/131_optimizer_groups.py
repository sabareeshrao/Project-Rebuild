"""Lesson 238: inspect which GPT tensors receive weight decay."""

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizer_parity import parameter_rank_groups


def main() -> None:
    model = GPTLanguageModel(GPTConfig(
        17, 8, 16, 4, 2, attention_impl="fused",
    ))
    report = parameter_rank_groups(model)
    print("unique tensor count:", report["unique_parameter_tensors"])
    print("trainable scalar parameters:", report["total_elements"])
    for group in ("decay", "no_decay"):
        print(group, "tensors:", len(report[group]))
        for name in report[group]:
            print(" ", name)


if __name__ == "__main__":
    main()
