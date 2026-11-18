import pytest
import torch

from rebuild_gpt.tensors import token_batch


def test_token_batch_has_batch_and_time_axes():
    batch = token_batch([[1, 2, 3], [4, 5, 6]])
    assert batch.shape == (2, 3)
    assert batch.dtype == torch.long
    assert batch.tolist() == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize("rows", [[], [1, 2, 3]])
def test_token_batch_rejects_missing_axis(rows):
    with pytest.raises(ValueError):
        token_batch(rows)
