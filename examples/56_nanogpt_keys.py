"""Lesson 56: map our readable fused-GPT tensor keys to nanoGPT-style names."""

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict

model = GPTLanguageModel(GPTConfig(9, 5, 8, 2, 2, attention_impl="fused"))
converted = export_nanogpt_state_dict(model)
for name, weight in list(converted.items())[:12]:
    print(f"{name:<45} {tuple(weight.shape)}")
print("upstream-style named tensors:", len(converted))
print("Only names/layouts are translated; an upstream checkpoint hasn't been loaded.")
