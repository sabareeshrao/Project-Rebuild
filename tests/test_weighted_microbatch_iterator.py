"""A lazy iterator makes no eager copies and weights uneven chunks correctly."""

import pytest
import torch

from rebuild_gpt.microbatches import iter_weighted_microbatches


def test_uneven_weights_equal_fraction_of_targets():
    tokens = torch.arange(30).reshape(5, 6)
    targets = torch.arange(25).reshape(5, 5)
    result = list(iter_weighted_microbatches(tokens, targets, microbatch_size=2))
    assert [x.shape[0] for x, _, _ in result] == [2, 2, 1]
    assert [weight for _, _, weight in result] == pytest.approx([0.4, 0.4, 0.2])
    assert sum(weight for _, _, weight in result) == pytest.approx(1.0)
    assert torch.equal(torch.cat([x for x, _, _ in result]), tokens)
    assert torch.equal(torch.cat([y for _, y, _ in result]), targets)
    assert result[0][0].data_ptr() == tokens.data_ptr()
    assert result[0][1].data_ptr() == targets.data_ptr()


def test_iterator_does_not_copy_or_move_tensors():
    inputs = torch.arange(12).reshape(3, 4)
    labels = torch.arange(3)
    iterator = iter_weighted_microbatches(inputs, labels, microbatch_size=2)
    assert iter(iterator) is iterator
    first_x, first_y, weight = next(iterator)
    assert first_x.device.type == "cpu"
    assert first_x.untyped_storage().data_ptr() == inputs.untyped_storage().data_ptr()
    assert first_y.untyped_storage().data_ptr() == labels.untyped_storage().data_ptr()
    assert weight == pytest.approx(2 / 3)
    assert list(iterator)[0][2] == pytest.approx(1 / 3)


def test_single_chunk_has_unit_weight():
    x = torch.ones(3, 4)
    y = torch.ones(3, dtype=torch.long)
    assert list(iter_weighted_microbatches(x, y, microbatch_size=9))[0][2] == 1.0


@pytest.mark.parametrize("x,y,micro", [
    (torch.empty(0, 3), torch.empty(0, 3), 2),
    (torch.ones(2, 3), torch.ones(1, 3), 2),
    (torch.tensor(1), torch.tensor(1), 2),
    (torch.ones(3, 3), torch.ones(3, 3), 0),
])
def test_bad_input_is_rejected_at_iteration(x, y, micro):
    with pytest.raises(ValueError):
        list(iter_weighted_microbatches(x, y, microbatch_size=micro))
