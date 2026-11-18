import pytest
import torch

from rebuild_gpt.tensors import next_token_pairs, token_batch


def test_token_batch_has_batch_and_time_axes():
    batch = token_batch([[1, 2, 3], [4, 5, 6]])
    assert batch.shape == (2, 3)
    assert batch.dtype == torch.long
    assert batch.tolist() == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize("rows", [[], [1, 2, 3]])
def test_token_batch_rejects_missing_axis(rows):
    with pytest.raises(ValueError):
        token_batch(rows)


def test_next_token_pairs_shift_targets_one_position():
    inputs, targets = next_token_pairs(token_batch([[10, 20, 30, 40]]))
    assert inputs.tolist() == [[10, 20, 30]]
    assert targets.tolist() == [[20, 30, 40]]


def test_next_token_pairs_rejects_single_token_context():
    with pytest.raises(ValueError):
        next_token_pairs(token_batch([[10]]))
