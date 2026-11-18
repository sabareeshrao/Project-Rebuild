"""Lesson 55: restart AdamW training with optimizer state and RNG restored."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.training import train_step
from rebuild_gpt.training_checkpoint import load_training_checkpoint, save_training_checkpoint

torch.manual_seed(19)
model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, dropout=0.2, attention_impl="fused"))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
x = torch.tensor([[0, 1, 2], [2, 3, 4]])
y = torch.tensor([[1, 2, 3], [3, 4, 5]])

for _ in range(2):
    train_step(model, optimizer, x, y)

with TemporaryDirectory() as directory:
    file = Path(directory) / "resume.pt"
    save_training_checkpoint(model, optimizer, step=2, path=file)
    continuous_loss = train_step(model, optimizer, x, y)
    restored, restored_optimizer, step, _ = load_training_checkpoint(file, restore_rng=True)
    restored_loss = train_step(restored, restored_optimizer, x, y)
    print("saved step:", step)
    print("uninterrupted loss:", continuous_loss)
    print("resumed loss:      ", restored_loss)
    print("same next step:", abs(continuous_loss - restored_loss) < 1e-7)
