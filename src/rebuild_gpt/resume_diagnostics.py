"""Diagnose the first meaningful difference between two trusted resume files."""

from pathlib import Path

import torch


def first_state_difference(left, right, location: str = "checkpoint") -> str | None:
    """Return the first nested field that differs (or None if identical)."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
                or left.dtype != right.dtype or left.shape != right.shape
                or not torch.equal(left, right)):
            return location
        return None
    if type(left) is not type(right):
        return location
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{location}.keys"
        for key in sorted(left, key=str):
            difference = first_state_difference(
                left[key], right[key], f"{location}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return f"{location}.length"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_state_difference(a, b, f"{location}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if left == right else location


def diagnose_resume_difference(left_path: str | Path, right_path: str | Path) -> dict:
    """Inspect differences without changing either checkpoint or global RNG.

    Only inspect trusted .pt files. weights_only=True reduces the attack
    surface but does not make untrusted downloaded checkpoints safe.
    """
    a = torch.load(left_path, map_location="cpu", weights_only=True)
    b = torch.load(right_path, map_location="cpu", weights_only=True)
    difference = first_state_difference(a, b)
    return {
        "identical": difference is None,
        "first_difference": difference,
    }
