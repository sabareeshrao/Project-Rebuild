"""Lesson 48: input embeddings and output scores share one learnable matrix."""

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.parameters import count_parameters

tied = GPTLanguageModel(GPTConfig(30, 16, 12, 3, 2))
untied = GPTLanguageModel(GPTConfig(30, 16, 12, 3, 2, tie_embeddings=False))

print("shared tensor:", tied.tokens.lookup.weight is tied.head.linear.weight)
print("tied parameters:", count_parameters(tied))
print("untied parameters:", count_parameters(untied))
print("parameters saved:", count_parameters(untied) - count_parameters(tied))
