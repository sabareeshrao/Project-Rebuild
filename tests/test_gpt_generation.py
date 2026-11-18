import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


def make_model():
    torch.manual_seed(5)
    return GPTLanguageModel(GPTConfig(9, 8, 8, 2, 1, attention_impl="fused")).eval()


def test_generation_preserves_prompt_and_appends_tokens():
    model = make_model()
    prompt = torch.tensor([[1, 2], [3, 4]])
    out = model.generate(prompt, 4, generator=torch.Generator().manual_seed(8))
    assert out.shape == (2, 6)
    assert torch.equal(out[:, :2], prompt)
    assert torch.all((out >= 0) & (out < 9))
    assert not out.requires_grad


def test_generation_seed_reproduces_sequence_without_changing_weights():
    model = make_model()
    prompt = torch.tensor([[1, 2]])
    a = model.generate(prompt, 5, generator=torch.Generator().manual_seed(17))
    b = model.generate(prompt, 5, generator=torch.Generator().manual_seed(17))
    assert torch.equal(a, b)


def test_zero_steps_returns_prompt_without_changing_it():
    model = make_model()
    prompt = torch.tensor([[1, 2]])
    out = model.generate(prompt, 0)
    assert torch.equal(out, prompt)


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_generation_rejects_invalid_step_count(steps):
    with pytest.raises(ValueError, match="max_new_tokens"):
        make_model().generate(torch.tensor([[1]]), steps)


def test_generation_validates_input_even_when_no_tokens_requested():
    model = make_model()
    with pytest.raises(ValueError, match="non-empty"):
        model.generate(torch.tensor([], dtype=torch.long), 0)
    with pytest.raises(ValueError, match="outside vocabulary"):
        model.generate(torch.tensor([[99]]), 0)


def test_generation_crops_long_prompt_but_preserves_full_output():
    model = make_model()
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1]])
    seed = 31
    full = model.generate(prompt, 4, generator=torch.Generator().manual_seed(seed))
    suffix = model.generate(prompt[:, -8:], 4, generator=torch.Generator().manual_seed(seed))
    assert full.shape == (1, 15)
    assert torch.equal(full[:, :11], prompt)
    assert torch.equal(full[:, 11:], suffix[:, 8:])


def test_generation_passes_at_most_block_size_tokens_to_model():
    model = make_model()
    observed = []
    original = model.forward_nanogpt

    def record_context(tokens, targets=None):
        observed.append(tokens.size(1))
        return original(tokens, targets)

    model.forward_nanogpt = record_context
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0]])
    model.generate(prompt, 3, generator=torch.Generator().manual_seed(3))
    assert observed == [8, 8, 8]


def test_generation_fills_short_prompt_before_cropping():
    model = make_model()
    observed = []
    original = model.forward_nanogpt

    def record_context(tokens, targets=None):
        observed.append(tokens.size(1))
        return original(tokens, targets)

    model.forward_nanogpt = record_context
    model.generate(torch.tensor([[0, 1, 2, 3, 4, 5, 6]]), 4)
    assert observed == [7, 8, 8, 8]


def test_temperature_validation_happens_even_with_zero_generation_steps():
    model = make_model()
    with pytest.raises(ValueError, match="temperature"):
        model.generate(torch.tensor([[1, 2]]), 0, temperature=0.0)


def test_seeded_sampling_with_temperature_is_reproducible():
    model = make_model()
    x = torch.tensor([[0, 1]])
    a = model.generate(x, 5, temperature=0.7, generator=torch.Generator().manual_seed(26))
    b = model.generate(x, 5, temperature=0.7, generator=torch.Generator().manual_seed(26))
    assert torch.equal(a, b)
