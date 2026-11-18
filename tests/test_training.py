import torch

from rebuild_gpt.bigram import BigramLanguageModel
from rebuild_gpt.training import train_step


def test_training_step_updates_bigram_weights():
    torch.manual_seed(12)
    model = BigramLanguageModel(vocab_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    inputs = torch.tensor([[0, 1, 0, 1]])
    targets = torch.tensor([[1, 0, 1, 0]])
    old = model.token_to_logits.weight.detach().clone()

    loss = train_step(model, optimizer, inputs, targets)

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.equal(old, model.token_to_logits.weight)


def test_repeated_training_reduces_loss_on_fixed_bigram_pattern():
    torch.manual_seed(2)
    model = BigramLanguageModel(vocab_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    inputs = torch.tensor([[0, 1, 0, 1]])
    targets = torch.tensor([[1, 0, 1, 0]])
    before = model.loss(inputs, targets).item()

    for _ in range(40):
        train_step(model, optimizer, inputs, targets)

    after = model.loss(inputs, targets).item()
    assert after < before
