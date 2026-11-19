"""Inspect a trusted character-GPT resume checkpoint without restoring it."""

from pathlib import Path

import torch

from rebuild_gpt.text_resume import TEXT_TRAINING_STATE_VERSION


def inspect_resume_checkpoint(path: str | Path) -> dict[str, int | str]:
    """Validate saved AdamW moments, training step and two CPU RNG streams.

    This read-only helper reuses the existing training-state format.
    Use it only with checkpoints from a trusted source.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format_version") != TEXT_TRAINING_STATE_VERSION:
        raise ValueError("Unsupported training-state format")
    step = state.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint step must be nonnegative")
    optimizer = state.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("Checkpoint is missing optimizer state")
    groups = optimizer.get("param_groups")
    moments = optimizer.get("state")
    if not isinstance(groups, list) or not groups or not isinstance(moments, dict):
        raise ValueError("Optimizer must contain parameter groups and state")

    ids = [parameter for group in groups for parameter in group["params"]]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Optimizer parameters must be unique and non-empty")
    if step > 0 and set(ids) != set(moments):
        raise ValueError("Optimizer moments do not cover all trainable parameters")
    for parameter_id, slots in moments.items():
        if not isinstance(slots, dict):
            raise ValueError(f"Invalid optimizer moment record for {parameter_id}")
        for name in ("step", "exp_avg", "exp_avg_sq"):
            value = slots.get(name)
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"Missing or nonfinite AdamW {name}")
        if float(slots["step"].item()) != step:
            raise ValueError("AdamW step counter does not match checkpoint step")
    for key in ("cpu_rng_state", "batch_generator_state"):
        rng = state.get(key)
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
            raise ValueError(f"Missing or malformed {key}")
    if not isinstance(state.get("model"), dict) or not isinstance(state.get("config"), dict):
        raise ValueError("Checkpoint lacks model or configuration")
    if not isinstance(state.get("corpus_sha256"), str):
        raise ValueError("Checkpoint lacks corpus digest")
    return {
        "step": step,
        "optimizer_groups": len(groups),
        "optimizer_parameters": len(ids),
        "moment_parameters": len(moments),
        "model_state_entries": len(state["model"]),
        "cpu_rng_bytes": state["cpu_rng_state"].numel(),
        "batch_rng_bytes": state["batch_generator_state"].numel(),
        "corpus_sha256": state["corpus_sha256"],
    }
