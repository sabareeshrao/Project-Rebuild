import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict
from rebuild_gpt.parameters import count_parameters


def test_bias_free_fused_gpt_has_upstream_compatible_tensor_shapes():
    cfg = GPTConfig(13, 5, 8, 2, 2, attention_impl="fused", bias=False)
    model = GPTLanguageModel(cfg)
    for block in model.blocks.blocks:
        assert block.ln1.bias is None and block.ln2.bias is None
        assert block.attention.qkv.c_attn.bias is None
        assert block.attention.c_proj.bias is None
        assert block.mlp.expand.bias is None and block.mlp.project.bias is None
    assert model.final_norm.bias is None
    c, v, t, layer = cfg.n_embd, cfg.vocab_size, cfg.block_size, cfg.n_layer
    assert count_parameters(model) == v*c + t*c + c + layer*(12*c*c + 2*c)
    assert all(not k.endswith(".bias") for k in export_nanogpt_state_dict(model))
    x = torch.tensor([[0, 1, 2]])
    model.loss(x, torch.tensor([[1, 2, 3]])).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_default_bias_configuration_is_unchanged():
    cfg = GPTConfig(7, 4, 8, 2, 1, attention_impl="fused")
    assert cfg.bias
    assert GPTLanguageModel(cfg).blocks.blocks[0].attention.c_proj.bias is not None


def test_bias_flag_requires_bool_and_fused_path():
    with pytest.raises(ValueError, match="bias must be a boolean"):
        GPTConfig(7, 4, 8, 2, 1, bias=0)
    with pytest.raises(ValueError, match="requires fused"):
        GPTConfig(7, 4, 8, 2, 1, bias=False)
