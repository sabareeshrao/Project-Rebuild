"""Lesson 75: keep the weights that achieved the best validation score."""

import torch
from torch import nn

from rebuild_gpt.best_model import BestValidationModel

torch.manual_seed(7)
model = nn.Linear(2, 2)
best = BestValidationModel()
best.consider(model, step=0, validation_loss=1.1)
old_weight = model.weight.detach().clone()

with torch.no_grad():
    model.weight.add_(10)
best.consider(model, step=10, validation_loss=1.3)
best.restore(model)

print("best step:", best.best_step)
print("best validation loss:", best.best_loss)
print("weights restored:", torch.equal(model.weight, old_weight))
