"""Lesson 54: save architecture and weights, then reload identical predictions."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rebuild_gpt.checkpoint import load_model_checkpoint, save_model_checkpoint
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel

torch.manual_seed(9)
model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 2, attention_impl="fused"))
model.eval()
input_ids = torch.tensor([[1, 3, 4]])

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "checkpoint.pt"
    save_model_checkpoint(model, path)
    reloaded = load_model_checkpoint(path)
    with torch.no_grad():
        print("same logits after reload:", torch.equal(model(input_ids), reloaded(input_ids)))
    print("tied weights preserved:", reloaded.tokens.lookup.weight is reloaded.head.linear.weight)
