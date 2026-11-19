"""Prove that stopping and resuming grouped AdamW training preserves a run."""

from pathlib import Path

from rebuild_gpt.resume_comparison import compare_resume_checkpoints
from rebuild_gpt.text_training import train_char_gpt


def run_resume_equivalence(
    corpus_path: str | Path,
    output_dir: str | Path,
    *,
    first_steps: int = 2,
    total_steps: int = 4,
    seed: int = 13,
) -> dict:
    """Compare full vs. split+resumed training with identical CPU controls.

    A small model and an evaluation at every step remove differences in
    evaluation cadence between the interrupted and uninterrupted runs.
    """
    if type(first_steps) is not int or first_steps < 1:
        raise ValueError("first_steps must be a positive integer")
    if type(total_steps) is not int or total_steps <= first_steps:
        raise ValueError("total_steps must exceed first_steps")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full_path = output / "full.pt"
    partial_path = output / "partial.pt"
    resumed_path = output / "resumed.pt"
    settings = {
        "block_size": 4,
        "n_embd": 8,
        "n_head": 2,
        "n_layer": 1,
        "batch_size": 2,
        "learning_rate": 0.01,
        "seed": seed,
        "device": "cpu",
        "grouped_adamw": True,
        "weight_decay": 0.1,
        "adam_betas": (0.9, 0.95),
        "eval_interval": 1,
        "eval_batches": 1,
        "warmup_steps": 1,
        "lr_decay_steps": total_steps,
        "min_lr": 0.001,
    }

    full_history: list[dict] = []
    _, full_tokenizer, full_metrics = train_char_gpt(
        corpus_path, steps=total_steps, history=full_history,
        training_state_path=full_path, **settings,
    )
    partial_history: list[dict] = []
    train_char_gpt(
        corpus_path, steps=first_steps, history=partial_history,
        training_state_path=partial_path, **settings,
    )
    resumed_history: list[dict] = []
    _, resumed_tokenizer, resumed_metrics = train_char_gpt(
        corpus_path, steps=total_steps, history=resumed_history,
        resume_from=partial_path, training_state_path=resumed_path, **settings,
    )
    if full_tokenizer.vocab != resumed_tokenizer.vocab:
        raise AssertionError("Tokenizer vocabulary mismatch")
    if full_metrics != resumed_metrics:
        raise AssertionError("Final training metrics differ")
    if full_history != resumed_history:
        raise AssertionError("Evaluation histories differ")
    checkpoint_report = compare_resume_checkpoints(full_path, resumed_path)
    return {
        "status": "identical",
        "first_steps": first_steps,
        "total_steps": total_steps,
        "seed": seed,
        "full_checkpoint": str(full_path),
        "partial_checkpoint": str(partial_path),
        "resumed_checkpoint": str(resumed_path),
        "checkpoint_comparison": checkpoint_report,
        "final_metrics": full_metrics,
        "evaluation_records": len(full_history),
    }
