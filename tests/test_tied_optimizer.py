import torch
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


def test_tied_weight_remains_shared_after_optimizer_update():
    torch.manual_seed(7)
    m = GPTLanguageModel(GPTConfig(5, 3, 8, 2, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=0.01)
    old = m.tokens.lookup.weight.detach().clone()
    opt.zero_grad()
    m.loss(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]])).backward()
    opt.step()
    assert m.tokens.lookup.weight is m.head.linear.weight
    assert not torch.equal(m.tokens.lookup.weight, old)
