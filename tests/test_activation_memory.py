import pytest
import torch

from rebuild_gpt.activation_memory import estimate_attention_matrix


def test_explicit_attention_score_shape_and_bytes():
    result = estimate_attention_matrix(
        batch_size=2, n_head=4, context_length=8,
    )
    assert result.score_shape == (2, 4, 8, 8)
    assert result.score_elements == 512
    assert result.bytes_per_matrix == 512 * 4


def test_doubling_context_quadruples_one_score_matrix():
    small = estimate_attention_matrix(
        batch_size=2, n_head=3, context_length=4,
    )
    big = estimate_attention_matrix(
        batch_size=2, n_head=3, context_length=8,
    )
    assert big.bytes_per_matrix == 4 * small.bytes_per_matrix


def test_bf16_halves_single_matrix_storage():
    fp32 = estimate_attention_matrix(
        batch_size=2, n_head=2, context_length=16,
    )
    bf16 = estimate_attention_matrix(
        batch_size=2, n_head=2, context_length=16, dtype=torch.bfloat16,
    )
    assert bf16.bytes_per_matrix * 2 == fp32.bytes_per_matrix


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"n_head": -1}, {"context_length": 0},
    {"context_length": 2.0}, {"batch_size": True},
])
def test_dimensions_require_positive_integers(kwargs):
    inputs = {"batch_size": 1, "n_head": 2, "context_length": 3}
    inputs.update(kwargs)
    with pytest.raises(ValueError, match="positive integer"):
        estimate_attention_matrix(**inputs)


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool, "fp32", None])
def test_dtype_must_be_floating_point(dtype):
    with pytest.raises(ValueError, match="floating-point"):
        estimate_attention_matrix(
            batch_size=1, n_head=2, context_length=3, dtype=dtype,
        )
