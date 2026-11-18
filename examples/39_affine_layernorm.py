"""Lesson 39: learned scale and bias make normalization adaptable."""

import torch
from torch.nn import functional as F

from rebuild_gpt.normalization import FeatureLayerNorm

torch.manual_seed(5)
layer = FeatureLayerNorm(n_embd=4)
features = torch.randn(2, 3, 4, requires_grad=True)
out = layer(features)
expected = F.layer_norm(features, (4,), layer.weight, layer.bias)
print("output agrees with PyTorch:", torch.allclose(out, expected))
out.square().mean().backward()
print("scale gradients:", layer.weight.grad.tolist())
print("bias gradients:", layer.bias.grad.tolist())
