"""Compare one uninterrupted GPT training run with several real restarts."""

from pathlib import Path

from rebuild_gpt.resume_comparison import compare_resume_checkpoints
from rebuild_gpt.resume_milestones import training_milestones
from rebuild_gpt.text_training import train_char_gpt


def resume_lab_options(total_steps: int, seed: int) -> dict:
    """Share exactly the same model, AdamW, data and LR controls across runs."""
    if type(total_steps) is not int or total_steps < 2:
        raise ValueError("total_steps must be an integer of at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    return dict(
        block_size=4, n_embd=8, n_head=2, n_layer=1,
        batch_size=2, learning_rate=0.01, seed=seed,
        device="cpu", grouped_adamw=True, weight_decay=0.1,
        adam_betas=(0.9, 0.95), eval_interval=1, eval_batches=1,
        warmup_steps=1, lr_decay_steps=total_steps, min_lr=0.001,
    )


def compare_multistage_resume(
    corpus_path: str | Path,
    output_dir: str | Path,
    *,
    total_steps: int = 6,
    stops: tuple[int, ...] = (2, 4),
    seed: int = 13,
) -> dict:
    """Prove exact CPU fp32 checkpoint equality after multiple interruptions."""
    milestones = training_milestones(total_steps, stops)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full_file = output / "uninterrupted.pt"
    final_file = output / "multistage_final.pt"
    options = resume_lab_options(total_steps, seed)
    full_history: list[dict] = []
    _, full_tok, full_metrics = train_char_gpt(
        corpus_path, steps=total_steps, history=full_history,
        training_state_path=full_file, **options,
    )
    previous_checkpoint = None
    stage_files: list[str] = []
    stage_history: list[dict] = []
    stage_metrics: dict = {}
    stage_tok = None
    for step in milestones:
        destination = (
            final_file if step == total_steps
            else output / f"after_step_{step:04d}.pt"
        )
        stage_history = []
        _, stage_tok, stage_metrics = train_char_gpt(
            corpus_path, steps=step, history=stage_history,
            resume_from=previous_checkpoint,
            training_state_path=destination,
            **options,
        )
        previous_checkpoint = destination
        stage_files.append(str(destination))

    if stage_tok is None or stage_tok.vocab != full_tok.vocab:
        raise AssertionError("Tokenizers differ after multiple restarts")
    if full_metrics != stage_metrics:
        raise AssertionError("Final training metrics differ")
    if full_history != stage_history:
        raise AssertionError("Evaluation histories differ")
    comparison = compare_resume_checkpoints(full_file, final_file)
    return {
        "status": "identical",
        "total_steps": total_steps,
        "stops": list(stops),
        "seed": seed,
        "uninterrupted_checkpoint": str(full_file),
        "stage_checkpoints": stage_files,
        "checkpoint_comparison": comparison,
        "validation_metrics": full_metrics,
        "evaluation_records": len(full_history),
    }
