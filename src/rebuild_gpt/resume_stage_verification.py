"""Verify every intermediate saved checkpoint in a multi-restart run."""

from hashlib import sha256
from pathlib import Path

import torch

from rebuild_gpt.resume_comparison import compare_resume_checkpoints
from rebuild_gpt.resume_inspection import inspect_resume_checkpoint
from rebuild_gpt.resume_milestones import training_milestones
from rebuild_gpt.text_resume import corpus_digest


def verify_multistage_checkpoints(run: dict, corpus_path: str | Path) -> dict:
    """Check step boundaries, corpus identity, controls and history prefixes."""
    points = training_milestones(run["total_steps"], run["stops"])
    filenames = run["stage_checkpoints"]
    if len(filenames) != len(points):
        raise ValueError("Stage checkpoint count does not match milestones")
    digest = corpus_digest(corpus_path)
    entries: list[dict] = []
    original_config = None
    original_vocab = None
    original_controls = None
    previous_history: list | None = None

    for expected_step, filename in zip(points, filenames):
        info = inspect_resume_checkpoint(filename)
        state = torch.load(filename, map_location="cpu", weights_only=True)
        if info["step"] != expected_step:
            raise ValueError(f"Stage {filename} has unexpected step {info['step']}")
        if info["corpus_sha256"] != digest:
            raise ValueError(f"Stage {filename} has different corpus identity")
        if info["optimizer_groups"] != 2 or info["moment_parameters"] < 1:
            raise ValueError("Expected initialized grouped AdamW optimizer")
        if original_config is None:
            original_config = state["config"]
            original_vocab = state["vocab"]
            original_controls = state["controls"]
        elif (state["config"] != original_config
              or state["vocab"] != original_vocab
              or state["controls"] != original_controls):
            raise ValueError(f"Stage {filename} changes model or optimizer controls")
        history = state["history"]
        if not isinstance(history, list):
            raise ValueError("Checkpoint history must be a list")
        if previous_history is not None and history[:len(previous_history)] != previous_history:
            raise ValueError(f"Stage {filename} does not preserve previous history")
        previous_history = history
        entries.append({
            "step": expected_step,
            "sha256": sha256(Path(filename).read_bytes()).hexdigest(),
            "optimizer_parameters": info["optimizer_parameters"],
            "moment_parameters": info["moment_parameters"],
            "evaluation_records": len(history),
        })

    final_comparison = compare_resume_checkpoints(
        run["uninterrupted_checkpoint"], filenames[-1],
    )
    if not final_comparison["identical"]:
        raise AssertionError("Uninterrupted and final stage checkpoint states differ")
    return {
        "corpus_sha256": digest,
        "stages": entries,
        "final_comparison": final_comparison,
    }
