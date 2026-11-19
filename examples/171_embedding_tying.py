"""Lesson 278: observe one shared input/output GPT embedding weight."""

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.embedding_tie_check import check_embedding_tying
from rebuild_gpt.gpt import GPTLanguageModel


def main() -> None:
    model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    print(check_embedding_tying(model))


if __name__ == "__main__":
    main()
