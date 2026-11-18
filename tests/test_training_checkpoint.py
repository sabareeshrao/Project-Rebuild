import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.training import train_step
from rebuild_gpt.training_checkpoint import load_training_checkpoint, save_training_checkpoint


def test_restored_adamw_and_rng_continue_training_identically(tmp_path):
    torch.manual_seed(19)
    config = GPTConfig(7, 4, 8, 2, 1, dropout=0.2, attention_impl="fused")
    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[0, 1, 2], [2, 3, 4]])
    y = torch.tensor([[1, 2, 3], [3, 4, 5]])
    for _ in range(2):
        train_step(model, optimizer, x, y)
    path = tmp_path / "training.pt"
    save_training_checkpoint(model, optimizer, 2, path)
    expected_loss = train_step(model, optimizer, x, y)
    expected_weights = [p.detach().clone() for p in model.parameters()]
    resumed, resumed_opt, step, _ = load_training_checkpoint(path, restore_rng=True)
    actual_loss = train_step(resumed, resumed_opt, x, y)
    assert step == 2
    assert actual_loss == pytest.approx(expected_loss, abs=1e-7)
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(expected_weights, resumed.parameters()))


def test_batch_generator_resumes_random_sample_sequence(tmp_path):
    torch.manual_seed(7)
    model = GPTLanguageModel(GPTConfig(5, 4, 8, 2, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    generator = torch.Generator().manual_seed(30)
    torch.randint(0, 100, (5,), generator=generator)
    path = tmp_path / "training.pt"
    save_training_checkpoint(model, optimizer, 8, path, batch_generator=generator)
    expected = torch.randint(0, 100, (5,), generator=generator)
    _, _, step, restored = load_training_checkpoint(path)
    assert step == 8 and restored is not None
    assert torch.equal(expected, torch.randint(0, 100, (5,), generator=restored))


def test_training_checkpoint_rejects_negative_step(tmp_path):
    model = GPTLanguageModel(GPTConfig(5, 4, 8, 2, 1))
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="step"):
        save_training_checkpoint(model, optimizer, -1, tmp_path / "invalid.pt")
