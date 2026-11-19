"""Validate a sequence of interruption points for repeatable GPT training."""

from collections.abc import Sequence


def training_milestones(total_steps: int, stops: Sequence[int]) -> tuple[int, ...]:
    """Return strictly increasing interruption points followed by the final step.

    A stop is a *completed* optimizer step. It must precede total_steps.
    """
    if type(total_steps) is not int or total_steps < 2:
        raise ValueError("total_steps must be an integer of at least two")
    if isinstance(stops, (str, bytes)) or not isinstance(stops, (list, tuple)):
        raise ValueError("stops must be a list or tuple of step numbers")
    if not stops:
        raise ValueError("At least one interruption point is required")
    previous = 0
    for step in stops:
        if type(step) is not int or step <= previous or step >= total_steps:
            raise ValueError("stops must be strictly increasing integers before total_steps")
        previous = step
    return (*stops, total_steps)
