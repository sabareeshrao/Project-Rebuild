"""Lesson 53: inspect checkpoint keys, tensor shapes, and shared weights."""

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.state_inspection import shared_parameters, state_shapes

model = GPTLanguageModel(GPTConfig(9, 8, 16, 4, 2, attention_impl="fused"))
for name, shape in list(state_shapes(model).items())[:16]:
    print(f"{name:<55} {shape}")
print("shared parameters:", shared_parameters(model))
print("state_dict names may repeat tied parameters; optimizers count them once.")
