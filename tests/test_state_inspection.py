import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.state_inspection import shared_parameters, state_shapes


def test_state_dict_lists_real_fused_gpt_tensor_shapes():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 2, attention_impl="fused"))
    shapes = state_shapes(model)
    assert shapes["tokens.lookup.weight"] == (7, 8)
    assert shapes["head.linear.weight"] == (7, 8)
    assert shapes["blocks.blocks.0.attention.qkv.c_attn.weight"] == (24, 8)
    assert shapes["blocks.blocks.1.mlp.expand.weight"] == (32, 8)
    assert len(shapes) == len(model.state_dict())


def test_tied_names_share_a_parameter_not_just_tensor_values():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, attention_impl="fused"))
    groups = shared_parameters(model)
    assert ("tokens.lookup.weight", "head.linear.weight") in groups
    assert model.tokens.lookup.weight.data_ptr() == model.head.linear.weight.data_ptr()
    assert torch.equal(model.state_dict()["tokens.lookup.weight"], model.state_dict()["head.linear.weight"])


def test_untied_embeddings_have_no_aliases():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, tie_embeddings=False))
    assert shared_parameters(model) == []
