"""NanoGPT-style initialization without copying upstream implementation."""

import math

import torch
from torch import nn

from rebuild_gpt.normalization import FeatureLayerNorm


def initialize_gpt_weights(model: nn.Module, n_layer: int) -> None:
    """Initialize each unique weight once; scale residual output matrices."""
    if type(n_layer) is not int or n_layer < 1:
        raise ValueError("n_layer must be a positive integer")
    residual_std = 0.02 / math.sqrt(2 * n_layer)
    seen_weights: set[int] = set()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # The tied output head and token lookup point to one Parameter.
                if id(module.weight) not in seen_weights:
                    scale_residual = name.endswith(
                        ("attention.projection", "attention.c_proj", "mlp.project")
                    )
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std if scale_residual else 0.02)
                    seen_weights.add(id(module.weight))
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, FeatureLayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
