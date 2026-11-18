import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.corpus_evaluation import full_corpus_loss, full_train_val_loss
from rebuild_gpt.gpt import GPTLanguageModel


def model():
    torch.manual_seed(8)
    return GPTLanguageModel(GPTConfig(4, 5, 8, 2, 1, dropout=0.3, attention_impl="fused"))


def test_every_transition_is_counted_once_including_short_last_chunk():
    net = model().eval()
    tokens = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1])
    with torch.no_grad():
        total = 0.0
        for start in (0, 4, 8):
            stop = min(start + 4, len(tokens) - 1)
            x = tokens[start:stop][None]
            y = tokens[start + 1:stop + 1][None]
            total += net.loss(x, y).item() * (stop - start)
    expected = total / (len(tokens) - 1)
    assert full_corpus_loss(net, tokens, block_size=4) == pytest.approx(expected, abs=2e-6)


def test_full_evaluation_preserves_mode_rng_and_gradients():
    net = model().train()
    before = torch.get_rng_state().clone()
    tokens = torch.tensor([0, 1, 2, 3] * 4)
    a = full_corpus_loss(net, tokens, block_size=3)
    b = full_corpus_loss(net, tokens, block_size=3)
    assert a == b
    assert net.training
    assert torch.equal(before, torch.get_rng_state())
    assert all(p.grad is None for p in net.parameters())


def test_full_train_val_uses_entire_independent_streams():
    net = model()
    a = torch.tensor([0, 1, 2, 3, 0, 1])
    b = torch.tensor([3, 2, 1, 0, 3, 2])
    scores = full_train_val_loss(net, a, b, block_size=3)
    assert scores["train_loss"] == full_corpus_loss(net, a, block_size=3)
    assert scores["validation_loss"] == full_corpus_loss(net, b, block_size=3)


@pytest.mark.parametrize("tokens,block_size,match", [
    (torch.tensor([1]), 2, "tokens"),
    (torch.tensor([1, 2, 3]).float(), 2, "tokens"),
    (torch.tensor([1, 2, 3]), 0, "block_size"),
    (torch.tensor([1, 2, 3]), 6, "block_size"),
])
def test_full_evaluation_rejects_invalid_inputs(tokens, block_size, match):
    with pytest.raises(ValueError, match=match):
        full_corpus_loss(model(), tokens, block_size=block_size)
