"""Verify tokens/update and score storage without a GPU."""

import pytest
import torch

from rebuild_gpt.workload_scaling import estimate_workload


def test_tokens_per_update_include_sequential_accumulation():
    item = estimate_workload(
        batch_size=3, context_length=8, n_head=2, n_layer=4,
        gradient_accumulation_steps=5,
    )
    assert item.tokens_per_microbatch == 24
    assert item.tokens_per_optimizer_step == 120
    assert item.score_bytes_per_layer == 3 * 2 * 8 * 8 * 4
    assert item.sum_score_bytes_all_layers == item.score_bytes_per_layer * 4


def test_depth_changes_score_sum_but_not_tokens_per_step():
    common = dict(batch_size=2, context_length=4, n_head=4)
    shallow = estimate_workload(n_layer=1, **common)
    deep = estimate_workload(n_layer=3, **common)
    assert deep.sum_score_bytes_all_layers == shallow.sum_score_bytes_all_layers * 3
    assert deep.tokens_per_optimizer_step == shallow.tokens_per_optimizer_step


def test_accumulation_changes_tokens_per_update_not_per_microbatch_matrix():
    kwargs = dict(batch_size=2, context_length=16, n_head=4, n_layer=2)
    base = estimate_workload(gradient_accumulation_steps=1, **kwargs)
    more = estimate_workload(gradient_accumulation_steps=8, **kwargs)
    assert more.tokens_per_optimizer_step == base.tokens_per_optimizer_step * 8
    assert more.score_bytes_per_layer == base.score_bytes_per_layer


def test_bf16_halves_score_bytes_compared_to_fp32():
    shape = dict(batch_size=2, context_length=16, n_head=2, n_layer=2)
    fp32 = estimate_workload(dtype=torch.float32, **shape)
    bf16 = estimate_workload(dtype=torch.bfloat16, **shape)
    assert bf16.score_bytes_per_layer * 2 == fp32.score_bytes_per_layer


@pytest.mark.parametrize("invalid", [0, -1, 1.1, True, "2"])
def test_rejects_invalid_depth(invalid):
    with pytest.raises(ValueError, match="n_layer"):
        estimate_workload(
            batch_size=2, context_length=4, n_head=2, n_layer=invalid,
        )


@pytest.mark.parametrize("invalid", [0, -3, 2.2, True])
def test_rejects_invalid_accumulation(invalid):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        estimate_workload(
            batch_size=2, context_length=4, n_head=2, n_layer=2,
            gradient_accumulation_steps=invalid,
        )
