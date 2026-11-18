import math

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.initialization import initialize_gpt_weights


def test_gpt_initialization_keeps_weight_sharing_and_resets_biases():
    torch.manual_seed(5)
    model = GPTLanguageModel(GPTConfig(13, 4, 64, 4, 3))
    assert model.tokens.lookup.weight is model.head.linear.weight
    assert abs(model.tokens.lookup.weight.std().item() - 0.02) < 0.006
    assert all(torch.count_nonzero(m.bias) == 0 for m in model.modules()
               if isinstance(m, torch.nn.Linear) and m.bias is not None)
    assert all(torch.allclose(m.weight, torch.ones_like(m.weight))
               for m in model.modules() if m.__class__.__name__ == 'FeatureLayerNorm')


def test_residual_projections_use_depth_scaled_std():
    torch.manual_seed(5)
    layer_count = 4
    model = GPTLanguageModel(GPTConfig(30, 4, 64, 4, layer_count))
    block = model.blocks.blocks[0]
    expected = 0.02 / math.sqrt(2 * layer_count)
    for tensor in (block.attention.projection.weight, block.mlp.project.weight):
        assert abs(tensor.std().item() - expected) < 0.003
    assert abs(block.attention.heads.heads[0].attention_weights.projections.query.weight.std().item() - 0.02) < 0.005


def test_initialization_rejects_invalid_layer_count():
    model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="n_layer"):
        initialize_gpt_weights(model, 0)


def test_gpt_initialization_is_deterministic_for_seed():
    cfg = GPTConfig(8, 4, 8, 2, 2)
    torch.manual_seed(123)
    first = GPTLanguageModel(cfg)
    torch.manual_seed(123)
    second = GPTLanguageModel(cfg)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
