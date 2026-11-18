import torch

from rebuild_gpt.bigram import BigramLanguageModel
from rebuild_gpt.data import sample_batch, split_train_val
from rebuild_gpt.training import train_step


def test_bigram_learns_repeated_transitions_and_generates_valid_tokens():
    """A small, repeatable integration test from data to gradients to sampling."""
    torch.manual_seed(3)
    tokens = torch.tensor([0, 1] * 80, dtype=torch.long)
    train, val = split_train_val(tokens, val_fraction=0.2)
    model = BigramLanguageModel(vocab_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    train_rng = torch.Generator().manual_seed(10)
    val_x, val_y = sample_batch(
        val, batch_size=8, block_size=4, generator=torch.Generator().manual_seed(11)
    )
    before = model.loss(val_x, val_y).item()

    for _ in range(40):
        x, y = sample_batch(train, batch_size=8, block_size=4, generator=train_rng)
        train_step(model, optimizer, x, y)

    with torch.no_grad():
        after = model.loss(val_x, val_y).item()
    assert after < before

    generated = model.generate(
        torch.tensor([[0]]), 10, generator=torch.Generator().manual_seed(12)
    )
    assert generated.shape == (1, 11)
    assert ((generated >= 0) & (generated < 2)).all()
