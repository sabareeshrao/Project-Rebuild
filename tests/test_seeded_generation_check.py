"""Sampling must respect a private generator and keep prompt tokens intact."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.seeded_generation_check import check_seeded_generation


def _model():
    return GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, dropout=0.2, attention_impl="fused",
    ))


def test_repeatable_sampling_uses_only_private_seeded_generator():
    model = _model()
    model.train()
    torch.manual_seed(123)
    rng = torch.get_rng_state().clone()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    report = check_seeded_generation(model, prompt, new_tokens=5, seed=19)
    assert report["repeatable"] and report["global_rng_preserved"]
    assert len(report["sampled_token_ids"]) == 2
    assert report["sampled_token_ids"][0][:3] == [1, 2, 3]
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training


class NonRepeatableGPT(GPTLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    def generate(self, token_ids, max_new_tokens, **kwargs):
        self.calls += 1
        # Wrong: ignores generator, substitutes a counter-dependent ID.
        new_token = self.calls % self.config.vocab_size
        return torch.cat((
            token_ids,
            torch.full(
                (token_ids.size(0), max_new_tokens), new_token,
                dtype=torch.long, device=token_ids.device,
            ),
        ), dim=1)


def test_nondeterministic_generation_is_caught():
    model = NonRepeatableGPT(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    with pytest.raises(AssertionError, match="Seeded generation differs"):
        check_seeded_generation(model, torch.tensor([[1, 2]]))


@pytest.mark.parametrize("options", [
    {"seed": -1}, {"seed": True}, {"new_tokens": 0},
    {"new_tokens": 1.2},
])
def test_invalid_sampling_parameters_are_rejected(options):
    with pytest.raises(ValueError):
        check_seeded_generation(_model(), torch.tensor([[1, 2]]), **options)
