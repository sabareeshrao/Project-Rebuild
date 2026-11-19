"""Stratified mean reweights per-window scores by valid start counts."""

import pytest
import torch

from rebuild_gpt.start_uniform_estimator import estimate_start_uniform_score


def test_region_size_weights_differ_from_raw_sample_mean():
    # Twenty-one eligible starts split into populations 5,5,5,6.
    report = estimate_start_uniform_score(
        [0, 1, 5, 10, 15], [1.0, 1.0, 2.0, 3.0, 4.0],
        corpus_token_count=25, block_size=4, region_count=4,
    )
    assert report.total_eligible_starts == 21
    assert report.region_population_sizes == (5, 5, 5, 6)
    assert report.region_start_counts == (2, 1, 1, 1)
    assert report.region_mean_scores == (1, 2, 3, 4)
    assert sum(report.region_population_weights) == pytest.approx(1)
    assert report.sampled_window_mean == pytest.approx(11 / 5)
    assert report.start_uniform_estimate == pytest.approx(54 / 21)
    assert report.reweighting_difference == pytest.approx(54 / 21 - 11 / 5)


def test_equal_scores_yield_same_estimate_regardless_of_uneven_region_sizes():
    x = estimate_start_uniform_score(
        torch.tensor([0, 5, 10, 15], dtype=torch.long),
        [2.25] * 4, corpus_token_count=25,
        block_size=4, region_count=4,
    )
    assert x.sampled_window_mean == x.start_uniform_estimate == 2.25


def test_single_region_is_unweighted_sample_mean():
    report = estimate_start_uniform_score(
        [0, 1, 2], [1, 3, 5], corpus_token_count=30,
        block_size=3, region_count=1,
    )
    assert report.sampled_window_mean == 3
    assert report.start_uniform_estimate == 3


@pytest.mark.parametrize("positions,values,pattern", [
    ([], [], "nonempty"),
    ([0], [1, 2], "equally sized"),
    ([0, 5, 10], [1, 2, 3], "every start region"),
    ([0, 5, 10, 21], [1, 2, 3, 4], "starts contain invalid"),
    ([0, 5, 10, True], [1, 2, 3, 4], "starts contain invalid"),
    ([0, 5, 10, 15], [1, 2, float("nan"), 4], "finite"),
    ([0, 5, 10, 15], [1, 2, True, 4], "finite"),
])
def test_rejects_invalid_or_incomplete_region_samples(positions, values, pattern):
    with pytest.raises(ValueError, match=pattern):
        estimate_start_uniform_score(
            positions, values, corpus_token_count=25,
            block_size=4, region_count=4,
        )


def test_rejects_gpu_or_wrong_dtype_start_vectors():
    with pytest.raises(ValueError, match="CPU torch.long"):
        estimate_start_uniform_score(
            torch.tensor([0, 5, 10, 15], dtype=torch.int32),
            [1, 2, 3, 4], corpus_token_count=25,
            block_size=4, region_count=4,
        )
