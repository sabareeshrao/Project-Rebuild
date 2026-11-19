import pytest
import torch

from rebuild_gpt.scaling_grid import build_scaling_grid, parse_positive_sizes


def test_parser_keeps_order_and_trims_whitespace():
    assert parse_positive_sizes("8, 2,16", label="contexts") == (8, 2, 16)


@pytest.mark.parametrize("value", ["", " ", "2,2", "0", "-1", "2,", "a,3", None])
def test_parser_rejects_invalid_lists(value):
    with pytest.raises(ValueError, match="contexts"):
        parse_positive_sizes(value, label="contexts")


def test_grid_uses_product_order_and_computes_effective_tokens():
    points = build_scaling_grid(
        (1, 2), (4, 8), (1, 3),
        n_head=2, gradient_accumulation_steps=2,
    )
    assert len(points) == 8
    assert [(p.batch_size, p.context_length, p.n_layer) for p in points] == [
        (1, 4, 1), (1, 4, 3), (1, 8, 1), (1, 8, 3),
        (2, 4, 1), (2, 4, 3), (2, 8, 1), (2, 8, 3),
    ]
    assert points[-1].tokens_per_optimizer_step == 2 * 8 * 2


def test_grid_is_limited_before_allocating_many_objects():
    with pytest.raises(ValueError, match="exceeding"):
        build_scaling_grid((1, 2, 3), (4, 8, 16), (1, 2, 3), max_points=8)


@pytest.mark.parametrize("field,value", [
    ("batch_sizes", (0,)), ("context_lengths", (True,)),
    ("depths", (1, 1)), ("depths", [1, 2]), ("batch_sizes", ()),
])
def test_grid_rejects_invalid_shape_sets(field, value):
    arguments = dict(
        batch_sizes=(1,), context_lengths=(4,), depths=(2,),
    )
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        build_scaling_grid(**arguments)


def test_invalid_dtype_and_head_count_propagate():
    with pytest.raises(ValueError, match="dtype"):
        build_scaling_grid((1,), (4,), (2,), dtype=torch.int32)
    with pytest.raises(ValueError, match="n_head"):
        build_scaling_grid((1,), (4,), (2,), n_head=0)
