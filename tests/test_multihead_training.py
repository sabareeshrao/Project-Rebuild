import torch

from rebuild_gpt.multihead_model import MultiHeadLanguageModel
from rebuild_gpt.training import train_step


def test_multihead_learns_to_predict_from_the_previous_token():
    """Current token and absolute position are equal, but targets differ."""
    torch.manual_seed(7)
    model = MultiHeadLanguageModel(vocab_size=3, block_size=2, n_embd=8, n_head=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.04)
    inputs = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
    targets = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)
    before = model.loss(inputs, targets).item()
    for _ in range(120):
        train_step(model, optimizer, inputs, targets)
    model.eval()
    with torch.no_grad():
        after = model.loss(inputs, targets).item()
        predictions = model(inputs).argmax(dim=-1)
    assert after < before
    assert after < 0.1
    assert torch.equal(predictions, targets)
    assert predictions[0, 1] != predictions[-1, 1]


def test_trained_multihead_model_is_causal_for_a_shorter_prefix():
    torch.manual_seed(0)
    model = MultiHeadLanguageModel(vocab_size=4, block_size=5, n_embd=8, n_head=2)
    model.eval()
    first = torch.tensor([[0, 1, 2, 3]])
    change_future = torch.tensor([[0, 1, 2, 0]])
    with torch.no_grad():
        assert torch.allclose(model(first)[:, :3], model(change_future)[:, :3], atol=1e-6)
        assert torch.allclose(model(first)[:, :3], model(first[:, :3]), atol=1e-6)
