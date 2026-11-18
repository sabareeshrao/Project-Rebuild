import torch

from rebuild_gpt.single_head_model import SingleHeadLanguageModel


def test_single_head_model_shapes_and_all_gradients():
    torch.manual_seed(3)
    model = SingleHeadLanguageModel(vocab_size=5, block_size=4, n_embd=6, head_size=3)
    x = torch.randint(0, 5, (2, 4))
    y = torch.randint(0, 5, (2, 4))
    assert model(x).shape == (2, 4, 5)
    model.loss(x, y).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_single_head_model_uses_past_but_not_future_tokens():
    model = SingleHeadLanguageModel(2, 3, 2, 2)
    with torch.no_grad():
        model.tokens.lookup.weight.copy_(torch.tensor([[0., 0.], [2., 0.]]))
        model.positions.lookup.weight.zero_()
        model.attention.attention_weights.projections.query.weight.zero_()
        model.attention.attention_weights.projections.key.weight.zero_()
        model.attention.values.value.weight.copy_(torch.eye(2))
        model.output.weight.copy_(torch.eye(2))
        model.head.linear.weight.copy_(torch.eye(2))
    first = model(torch.tensor([[0, 1, 0]]))
    changed_past = model(torch.tensor([[1, 1, 0]]))
    changed_future = model(torch.tensor([[0, 1, 1]]))
    assert not torch.allclose(first[:, 1], changed_past[:, 1])
    assert torch.allclose(first[:, :2], changed_future[:, :2])
    assert torch.allclose(first[:, :2], model(torch.tensor([[0, 1]])))
