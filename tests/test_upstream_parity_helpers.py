import pytest
import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.upstream_parity import (
    load_upstream_model,
    synchronize_upstream_weights,
)


def test_missing_reference_file_reports_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_upstream_model(tmp_path / "missing.py", GPTConfig(7, 4, 8, 2, 1))


def test_reference_key_mismatch_is_rejected_instead_of_skipped():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, attention_impl="fused"))
    reference = nn.Linear(8, 7)
    with pytest.raises(ValueError, match="Reference key mismatch"):
        synchronize_upstream_weights(model, reference)


def test_reference_shape_mismatch_is_reported():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, attention_impl="fused"))
    state = model.state_dict()
    assert state["tokens.lookup.weight"].shape == (7, 8)
    assert model.tokens.lookup.weight is model.head.linear.weight


def test_educational_attention_is_not_offered_as_upstream_parity():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1))
    with pytest.raises(ValueError, match="fused attention"):
        synchronize_upstream_weights(model, nn.Linear(8, 7))
