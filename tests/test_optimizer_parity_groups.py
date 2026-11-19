"""Weight decay grouping must count tied GPT parameters once."""

import pytest
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizer_parity import parameter_rank_groups


def test_gpt_embeddings_and_layer_norm_are_in_correct_groups():
    model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 2, attention_impl="fused"))
    report = parameter_rank_groups(model)
    assert "tokens.lookup.weight" in report["decay"]
    assert "final_norm.weight" in report["no_decay"]
    assert "head.linear.weight" not in report["decay"]
    assert report["total_elements"] == sum(p.numel() for p in model.parameters())
    assert report["unique_parameter_tensors"] == len(list(model.parameters()))
    assert report["decay_elements"] > 0
    assert report["no_decay_elements"] > 0


def test_frozen_parameters_are_excluded():
    layer = nn.Linear(3, 2)
    layer.bias.requires_grad_(False)
    report = parameter_rank_groups(layer)
    assert report["decay"] == ["weight"]
    assert report["no_decay"] == []
    assert report["total_elements"] == 6


def test_empty_model_cannot_be_grouped():
    with pytest.raises(ValueError, match="no trainable"):
        parameter_rank_groups(nn.Identity())
