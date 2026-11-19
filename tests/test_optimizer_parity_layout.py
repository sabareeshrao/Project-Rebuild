"""Compare actual AdamW group membership, not only the grouping recipe."""

import pytest
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizer_parity import (
    compare_grouped_optimizer_layout, optimizer_decay_by_name,
)
from rebuild_gpt.optimizers import configure_adamw


class UpstreamNamedAdapter(nn.Module):
    """A local naming adapter for tests, NOT the real upstream model."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def named_parameters(self, *args, **kwargs):
        from rebuild_gpt.nanogpt_keys import upstream_key
        for name, param in self.model.named_parameters(*args, **kwargs):
            yield upstream_key(name), param


def test_equal_groups_match_regardless_of_name_mapping():
    model = GPTLanguageModel(GPTConfig(9, 4, 8, 2, 1, attention_impl="fused"))
    reference = UpstreamNamedAdapter(model)
    ours_opt = configure_adamw(model, learning_rate=0.001)
    their_opt = configure_adamw(reference, learning_rate=0.001)
    report = compare_grouped_optimizer_layout(
        model, reference, ours_opt, their_opt, weight_decay=0.1,
    )
    assert report["matched_parameters"] == len(list(model.parameters()))
    assert report["decay_parameters"] > 0
    assert report["no_decay_parameters"] > 0


def test_mismatch_in_actual_decay_configuration_is_reported():
    model = GPTLanguageModel(GPTConfig(9, 4, 8, 2, 1, attention_impl="fused"))
    ref = UpstreamNamedAdapter(model)
    ours_opt = configure_adamw(model, learning_rate=0.001, weight_decay=0.1)
    ref_opt = configure_adamw(ref, learning_rate=0.001, weight_decay=0.2)
    with pytest.raises(ValueError, match="weight decay mismatch"):
        compare_grouped_optimizer_layout(
            model, ref, ours_opt, ref_opt, weight_decay=0.1,
        )


def test_optimizer_parameter_coverage_is_checked():
    module = nn.Linear(3, 2)
    from torch.optim import AdamW
    opt = AdamW([module.weight])
    with pytest.raises(ValueError, match="all trainable"):
        optimizer_decay_by_name(module, opt)
