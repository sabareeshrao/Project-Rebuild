"""Resolve the execution device before allocating model parameters."""

import torch


def resolve_device(requested: str = "cpu") -> torch.device:
    """Choose CPU or one CUDA accelerator, with explicit availability errors.

    'auto' prefers CUDA when available. Explicit 'cuda' must never silently
    fall back to CPU, since that would make performance results misleading.
    """
    if requested not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    chosen = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if chosen == "auto":
        chosen = "cpu"
    if chosen == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no CUDA accelerator is available")
    return torch.device(chosen)
