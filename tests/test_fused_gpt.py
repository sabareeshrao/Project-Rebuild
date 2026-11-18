import math

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.fused_attention import FusedCausalSelfAttention
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.parameters import count_parameters
from rebuild_gpt.training import train_step


def test_fused_gpt_has_attention_and_correct_parameter_count():
    cfg = GPTConfig(7, 4, 8, 2, 2, attention_impl="fused")
    model = GPTLanguageModel(cfg)
    assert all(isinstance(b.attention, FusedCausalSelfAttention) for b in model.blocks.blocks)
    c, v, t, layers = cfg.n_embd, cfg.vocab_size, cfg.block_size, cfg.n_layer
    assert count_parameters(model) == v*c + t*c + 2*c + layers*(12*c*c + 13*c)
    assert model.tokens.lookup.weight is model.head.linear.weight
    x = torch.tensor([[0, 1, 2, 3]])
    loss = model.loss(x, torch.tensor([[1, 2, 3, 4]]))
    assert model(x).shape == (1, 4, 7)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_residual_initialization_is_scaled_in_fused_attention():
    torch.manual_seed(4)
    model = GPTLanguageModel(GPTConfig(7, 4, 64, 4, 4, attention_impl="fused"))
    expected = 0.02 / math.sqrt(8)
    block = model.blocks.blocks[0]
    assert abs(block.attention.c_proj.weight.std().item() - expected) < 0.003
    assert abs(block.attention.qkv.c_attn.weight.std().item() - 0.02) < 0.004


def test_fused_gpt_outputs_are_causal_across_multiple_layers():
    torch.manual_seed(7)
    model = GPTLanguageModel(GPTConfig(6, 5, 12, 3, 3, attention_impl="fused"))
    model.eval()
    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    changed = torch.tensor([[0, 1, 2, 5, 5]])
    with torch.no_grad():
        assert torch.allclose(model(tokens)[:, :3], model(changed)[:, :3], atol=1e-5)
        assert torch.allclose(model(tokens)[:, :3], model(tokens[:, :3]), atol=1e-5)


def test_fused_gpt_trains_on_context_dependent_targets():
    torch.manual_seed(7)
    model = GPTLanguageModel(GPTConfig(3, 2, 8, 2, 2, attention_impl="fused"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025)
    inputs = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
    targets = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)
    before = model.loss(inputs, targets).item()
    for _ in range(120):
        train_step(model, optimizer, inputs, targets)
    model.eval()
    with torch.no_grad():
        after = model.loss(inputs, targets).item()
        predicted = model(torch.tensor([[0, 1], [2, 1]])).argmax(-1)
    assert after < before
    assert after < 0.12
    assert torch.equal(predicted, torch.tensor([[1, 0], [1, 2]]))


def test_both_attention_implementations_keep_same_output_shape():
    x = torch.tensor([[0, 1, 2]])
    for variant in ("educational", "fused"):
        model = GPTLanguageModel(GPTConfig(5, 3, 8, 2, 1, attention_impl=variant))
        assert model(x).shape == (1, 3, 5)
