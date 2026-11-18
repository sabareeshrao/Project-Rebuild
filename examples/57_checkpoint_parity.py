"""Lesson 57: compare predictions before/after saving and key translation."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rebuild_gpt.checkpoint import load_model_checkpoint, save_model_checkpoint
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict, load_nanogpt_state_dict
from rebuild_gpt.training import train_step

torch.manual_seed(23)
config = GPTConfig(11, 5, 8, 2, 2, attention_impl="fused")
model = GPTLanguageModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
inputs = torch.tensor([[0, 1, 2, 3]])
targets = torch.tensor([[1, 2, 3, 4]])
train_step(model, optimizer, inputs, targets)
model.eval()

with TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "gpt.pt"
    save_model_checkpoint(model, checkpoint)
    loaded = load_model_checkpoint(checkpoint)
    mapped = export_nanogpt_state_dict(loaded)
    imported = GPTLanguageModel(config).eval()
    load_nanogpt_state_dict(imported, mapped)
    with torch.no_grad():
        print("native restore identical logits:", torch.equal(model(inputs), loaded(inputs)))
        print("translated restore identical logits:", torch.equal(model(inputs), imported(inputs)))
    print("mapped tensor count:", len(mapped))
    print("Note: a real upstream nanoGPT checkpoint has NOT been verified here.")
