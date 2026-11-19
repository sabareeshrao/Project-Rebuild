"""Apply a fresh, non-persistent optimizer-parity experiment to a capstone."""

from pathlib import Path

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.training_parity import compare_one_adamw_step
from rebuild_gpt.upstream_parity import load_upstream_model


def compare_saved_capstone_adamw(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    upstream_model_path: str | Path,
    *,
    batch_size: int = 2,
    seed: int = 29,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
) -> dict[str, int | float]:
    """Load independent in-memory models; leave the checkpoint unchanged.

    Importing upstream_model_path EXECUTES code from that file, so callers
    must supply a trusted, reviewed local nanoGPT model.py.
    """
    ours, tokenizer = load_text_checkpoint(checkpoint_path)
    inputs, targets = prepare_validation_parity_batch(
        corpus_path, tokenizer, ours.config,
        batch_size=batch_size, seed=seed,
    )
    upstream = load_upstream_model(upstream_model_path, ours.config)
    result = compare_one_adamw_step(
        ours, upstream, inputs, targets,
        learning_rate=learning_rate, weight_decay=weight_decay,
    )
    return {
        "batch_size": inputs.shape[0],
        "context_length": inputs.shape[1],
        "seed": seed,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "matched_gradients": result.matched_gradients,
        "max_gradient_error": result.max_gradient_error,
        "matched_parameter_tensors": result.matched_parameter_tensors,
        "max_updated_weight_error": result.max_updated_weight_error,
        "largest_local_parameter_change": result.largest_local_parameter_change,
    }
