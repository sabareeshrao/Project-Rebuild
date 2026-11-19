"""Long prompts must be preserved even though attention sees a short window."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.generation_context_check import check_generation_context
from rebuild_gpt.gpt import GPTLanguageModel


def _model(cls=GPTLanguageModel):
    return cls(GPTConfig(
        11, 4, 8, 2, 1, dropout=0.2, attention_impl="fused",
    ))


def _prompt():
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7],
                         [7, 6, 5, 4, 3, 2, 1]])


def test_generate_matches_manual_cropping_on_long_prompts():
    model = _model()
    model.train()
    before_rng = torch.get_rng_state().clone()
    report = check_generation_context(model, _prompt(), new_tokens=5, seed=19)
    assert report["status"] == "matched"
    assert report["prompt_tokens"] == 7
    assert report["block_size"] == 4
    assert len(report["generated_ids"][0]) == 12
    assert report["generated_ids"][0][:7] == [1, 2, 3, 4, 5, 6, 7]
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert model.training


class DropsPromptPrefixGPT(GPTLanguageModel):
    def generate(self, token_ids, max_new_tokens, **kwargs):
        # Wrong: truncates the RETURNED sequence, not only the model context.
        return super().generate(
            token_ids[:, -self.config.block_size:], max_new_tokens, **kwargs,
        )


def test_bad_generator_that_discards_prompt_is_caught():
    with pytest.raises(AssertionError, match="output shape"):
        check_generation_context(_model(DropsPromptPrefixGPT), _prompt())


@pytest.mark.parametrize("prompt,new_tokens", [
    (torch.tensor([[1, 2, 3]]), 3),
    (torch.tensor([[1., 2., 3., 4., 5.]]), 3),
    (torch.tensor([[1, 2, 3, 4, 99]]), 3),
    (torch.tensor([[1, 2, 3, 4, 5]]), 0),
])
def test_invalid_inputs_fail_before_sampling(prompt, new_tokens):
    with pytest.raises(ValueError):
        check_generation_context(_model(), prompt, new_tokens=new_tokens)
