import torch
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.training import train_step


def test_stacked_model_learns_a_context_dependent_toy_pattern():
    torch.manual_seed(7)
    model = GPTLanguageModel(GPTConfig(vocab_size=3, block_size=2, n_embd=8, n_head=2, n_layer=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025)
    x = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
    y = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)
    before = model.loss(x, y).item()
    for _ in range(120):
        train_step(model, optimizer, x, y)
    model.eval()
    with torch.no_grad():
        after = model.loss(x, y).item()
        predictions = model(torch.tensor([[0, 1], [2, 1]])).argmax(dim=-1)
    assert after < before
    assert after < 0.12
    assert torch.equal(predictions, torch.tensor([[1, 0], [1, 2]]))
