"""Invalid restart schedules must fail before any expensive training."""

import pytest

from rebuild_gpt.resume_milestones import training_milestones


def test_two_interruption_points_form_three_training_segments():
    assert training_milestones(6, (2, 4)) == (2, 4, 6)


def test_different_segment_lengths_are_supported():
    assert training_milestones(8, [1, 3, 7]) == (1, 3, 7, 8)


@pytest.mark.parametrize("total,stops", [
    (1, [1]), (True, [1]), (6, []), (6, [0]),
    (6, [3, 3]), (6, [4, 2]), (6, [6]),
    (6, [True]), (6, [1.0]), (6, "2,4"),
])
def test_invalid_milestone_plan_is_rejected(total, stops):
    with pytest.raises(ValueError):
        training_milestones(total, stops)
