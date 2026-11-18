import pytest
import torch
from torch import nn

from rebuild_gpt.bigram import BigramLanguageModel


def test_bigram_logits_have_batch_time_vocab_shape():
    model = BigramLanguageModel(vocab_size=4)
    ids = torch.tensor([[0, 1, 2], [3, 2, 1]])
    logits = model(ids)
    assert logits.shape == (2, 3, 4)
    assert isinstance(model.token_to_logits, nn.Embedding)
    assert sum(parameter.numel() for parameter in model.parameters()) == 16


def test_same_current_token_has_identical_logits_regardless_of_context():
    model = BigramLanguageModel(vocab_size=4)
    logits = model(torch.tensor([[0, 2], [1, 2]]))
    assert torch.equal(logits[0, 1], logits[1, 1])


@pytest.mark.parametrize("vocab_size", [0, 1])
def test_bigram_requires_at_least_two_tokens(vocab_size):
    with pytest.raises(ValueError, match="at least 2"):
        BigramLanguageModel(vocab_size)


def test_bigram_rejects_invalid_token_ids():
    model = BigramLanguageModel(vocab_size=3)
    with pytest.raises(ValueError, match="long tensor"):
        model(torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="outside vocabulary"):
        model(torch.tensor([[3]]))


def test_bigram_loss_matches_reference_cross_entropy():
    model = BigramLanguageModel(vocab_size=3)
    x = torch.tensor([[0, 1], [2, 0]])
    y = torch.tensor([[1, 2], [0, 1]])
    expected = torch.nn.functional.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
    assert torch.allclose(model.loss(x, y), expected)


def test_bigram_loss_creates_embedding_gradients():
    model = BigramLanguageModel(vocab_size=3)
    x = torch.tensor([[0, 1, 2]])
    y = torch.tensor([[1, 2, 0]])
    model.loss(x, y).backward()
    grad = model.token_to_logits.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_generation_preserves_prompt_and_extends_both_rows():
    torch.manual_seed(4)
    model = BigramLanguageModel(vocab_size=3)
    prompt = torch.tensor([[0, 1], [2, 0]])
    generated = model.generate(prompt, num_new_tokens=5, generator=torch.Generator().manual_seed(8))
    assert generated.shape == (2, 7)
    assert torch.equal(generated[:, :2], prompt)
    assert ((0 <= generated) & (generated < 3)).all()


def test_generation_is_reproducible_for_same_seed():
    model = BigramLanguageModel(vocab_size=4)
    prompt = torch.tensor([[0, 1]])
    a = model.generate(prompt, 8, generator=torch.Generator().manual_seed(10))
    b = model.generate(prompt, 8, generator=torch.Generator().manual_seed(10))
    assert torch.equal(a, b)


def test_generation_depends_only_on_last_token_for_bigram():
    model = BigramLanguageModel(vocab_size=4)
    prompts = torch.tensor([[0, 2], [1, 2]])
    output = model.generate(prompts[:1], 6, generator=torch.Generator().manual_seed(5))
    other = model.generate(prompts[1:], 6, generator=torch.Generator().manual_seed(5))
    assert torch.equal(output[0, 2:], other[0, 2:])


def test_generation_rejects_invalid_prompt_or_negative_length():
    model = BigramLanguageModel(vocab_size=3)
    with pytest.raises(ValueError, match="nonnegative"):
        model.generate(torch.tensor([[0]]), -1)
    with pytest.raises(ValueError, match="non-empty"):
        model.generate(torch.tensor([], dtype=torch.long), 1)


def test_generation_does_not_track_gradients():
    model = BigramLanguageModel(vocab_size=3)
    out = model.generate(torch.tensor([[0]]), 3)
    assert not out.requires_grad
