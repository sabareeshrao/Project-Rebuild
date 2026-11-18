import torch

from rebuild_gpt.single_head_model import SingleHeadLanguageModel
from rebuild_gpt.training import train_step


def test_attention_model_learns_to_use_earlier_tokens():
    """At position 1, the same current token needs different answers."""
    torch.manual_seed(7)
    model = SingleHeadLanguageModel(vocab_size=3, block_size=2, n_embd=8, head_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.04)

    inputs = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
    targets = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)
    before = model.loss(inputs, targets).item()

    for _ in range(140):
        train_step(model, optimizer, inputs, targets)

    after = model.loss(inputs, targets).item()
    assert after < before
    assert after < 0.1
    assert torch.equal(model(inputs).argmax(dim=-1), targets)
