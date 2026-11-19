"""One logical-batch update equals one optimizer step regardless of chunks."""

import copy

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.training import train_logical_batch_step, train_step


def make_model():
    return GPTLanguageModel(GPTConfig(
        vocab_size=9, block_size=4, n_embd=8, n_head=2,
        n_layer=1, attention_impl="sdpa", dropout=0.0,
    ))


@pytest.mark.parametrize("microbatch_size", [1, 2, 3, 5, 10])
def test_logical_batch_gpt_matches_full_batch_update(microbatch_size):
    torch.manual_seed(138)
    full = make_model()
    chunked = copy.deepcopy(full)
    x = torch.tensor([
        [0, 1, 2], [2, 3, 4], [4, 5, 6], [5, 6, 7], [7, 8, 0],
    ])
    y = torch.tensor([
        [1, 2, 3], [3, 4, 5], [5, 6, 7], [6, 7, 8], [8, 0, 1],
    ])
    opt_full = torch.optim.AdamW(full.parameters(), lr=0.001)
    opt_part = torch.optim.AdamW(chunked.parameters(), lr=0.001)
    full_loss = train_step(full, opt_full, x, y)
    part_loss = train_logical_batch_step(
        chunked, opt_part, x, y, microbatch_size=microbatch_size,
    )
    assert part_loss == pytest.approx(full_loss, abs=3e-6)
    for a, b in zip(full.parameters(), chunked.parameters()):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
    assert all(int(state["step"].item()) == 1 for state in opt_part.state.values())


def test_invalid_chunk_size_fails_before_optimizer_update():
    torch.manual_seed(138)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters())
    x = torch.zeros(3, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="microbatch_size"):
        train_logical_batch_step(model, opt, x, x, microbatch_size=0)
    assert not opt.state


def test_bf16_logical_batch_leaves_parameters_fp32():
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.tensor([[0, 1, 2], [3, 4, 5], [5, 6, 7]])
    y = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
    loss = train_logical_batch_step(
        model, opt, x, y, microbatch_size=2, precision="bf16",
    )
    assert torch.isfinite(torch.tensor(loss))
    assert all(p.dtype == torch.float32 for p in model.parameters())
