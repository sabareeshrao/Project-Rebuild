"""Strictly compare two trusted CPU GPT training resume states."""

from pathlib import Path

import torch

from rebuild_gpt.resume_inspection import inspect_resume_checkpoint


def _assert_identical(left, right, path: str) -> int:
    """Return number of equal tensors; identify the first differing field."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError(f"{path}: tensor type mismatch")
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError(f"{path}: tensor values, shape or dtype differ")
        return 1
    if type(left) is not type(right):
        raise AssertionError(f"{path}: incompatible value types")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f"{path}: dictionary keys differ")
        return sum(
            _assert_identical(left[k], right[k], f"{path}.{k}")
            for k in sorted(left, key=str)
        )
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: sequence lengths differ")
        return sum(
            _assert_identical(a, b, f"{path}[{i}]")
            for i, (a, b) in enumerate(zip(left, right))
        )
    if left != right:
        raise AssertionError(f"{path}: values differ ({left!r} vs {right!r})")
    return 0


def compare_resume_checkpoints(left_path: str | Path, right_path: str | Path) -> dict:
    """Require bitwise-equal saved weights, optimizer, RNG and progress.

    Both inputs must be trusted checkpoints. Neither model nor optimizer
    object is instantiated, and no global RNG state is changed.
    """
    left_info = inspect_resume_checkpoint(left_path)
    right_info = inspect_resume_checkpoint(right_path)
    if left_info != right_info:
        raise AssertionError("Resume checkpoint metadata differ")

    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    fields = (
        "config", "vocab", "corpus_sha256", "controls", "step",
        "model", "optimizer", "cpu_rng_state", "batch_generator_state",
        "cuda_rng_state_all", "grad_scaler_state", "history",
        "validation_before", "last_loss", "last_learning_rate", "best_state",
    )
    count = sum(
        _assert_identical(left[key], right[key], key)
        for key in fields
    )
    return {
        "step": left_info["step"],
        "identical": True,
        "matched_tensors": count,
        "optimizer_parameters": left_info["optimizer_parameters"],
        "moment_parameters": left_info["moment_parameters"],
    }
