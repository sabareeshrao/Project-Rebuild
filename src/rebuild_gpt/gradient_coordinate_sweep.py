"""Numerically verify several GPT gradients in different model components."""

import torch

from rebuild_gpt.gradient_check import check_parameter_gradient
from rebuild_gpt.gpt import GPTLanguageModel


DEFAULT_COORDINATES = (
    ("tokens.lookup.weight", (0, 0)),
    ("tokens.lookup.weight", (1, 0)),
    ("positions.lookup.weight", (0, 0)),
)


def check_gradient_coordinates(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    coordinates: tuple = DEFAULT_COORDINATES,
) -> dict:
    """Run independent double-precision finite differences at several sites.

    Reuses the individual check, which restores each weight, clears
    gradients, and restores eval/train mode even if a check fails.
    """
    if (not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2
            or any(not isinstance(entry, tuple) or len(entry) != 2
                   or not isinstance(entry[0], str)
                   or not isinstance(entry[1], tuple)
                   for entry in coordinates)):
        raise ValueError("coordinates must provide at least two (parameter, index) pairs")
    if len({(name, index) for name, index in coordinates}) != len(coordinates):
        raise ValueError("Gradient coordinates must be distinct")
    results = []
    for name, index in coordinates:
        results.append(check_parameter_gradient(
            model, tokens, targets, parameter_name=name, index=index,
        ))
    return {
        "status": "passed",
        "checked_coordinates": len(results),
        "checked_parameter_names": sorted({row["parameter"] for row in results}),
        "max_absolute_error": max(row["absolute_error"] for row in results),
        "results": results,
    }
