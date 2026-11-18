import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.parameters import count_parameters


def test_model_parameter_count_matches_architecture_formula():
    config = GPTConfig(vocab_size=5, block_size=4, n_embd=8, n_head=2, n_layer=2)
    model = GPTLanguageModel(config)
    c, v, t, layers = config.n_embd, config.vocab_size, config.block_size, config.n_layer
    expected = 2 * v * c + t * c + 2 * c + layers * (12 * c * c + 10 * c)
    assert count_parameters(model) == expected
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())


def test_more_layers_increase_parameter_count_without_changing_shape():
    shallow = GPTLanguageModel(GPTConfig(5, 4, 8, 2, 1))
    deep = GPTLanguageModel(GPTConfig(5, 4, 8, 2, 3))
    x = torch.tensor([[0, 1, 2]])
    assert deep(x).shape == shallow(x).shape
    assert count_parameters(deep) > count_parameters(shallow)


def test_parameter_count_can_exclude_frozen_weights():
    model = nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    assert count_parameters(model) == 6
    assert count_parameters(model, trainable_only=False) == 8
