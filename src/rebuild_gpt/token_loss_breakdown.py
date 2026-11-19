"""Explain mean next-token loss as a weighted collection of token NLLs."""

import torch

from rebuild_gpt.perplexity_reference import compare_uniform_perplexity


def inspect_token_losses(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Preserve ignored positions as None and weight rows by valid tokens.

    A row with all ignored targets contributes zero supervised tokens.
    Its own average loss is undefined (None), not zero.
    """
    benchmark = compare_uniform_perplexity(logits, targets)
    safe_targets = targets.clamp_min(0)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    nll = -log_probabilities.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    mask = targets != -1
    per_position = [
        [float(nll[row, col]) if bool(mask[row, col]) else None
         for col in range(targets.size(1))]
        for row in range(targets.size(0))
    ]
    counts = [int(mask[row].sum()) for row in range(targets.size(0))]
    row_means = [
        float(nll[row][mask[row]].mean()) if counts[row] else None
        for row in range(targets.size(0))
    ]
    numerator = float(nll[mask].sum())
    independently_reconstructed = numerator / benchmark["supervised_positions"]
    expected = benchmark["cross_entropy_nats"]
    error = abs(independently_reconstructed - expected)
    if error > 2e-6:
        raise AssertionError("Token NLLs do not reconstruct the masked mean CE")
    return {
        "supervised_positions": benchmark["supervised_positions"],
        "ignored_positions": benchmark["ignored_positions"],
        "per_position_nll": per_position,
        "row_supervised_counts": counts,
        "row_mean_nll": row_means,
        "mean_nll": independently_reconstructed,
        "mean_cross_entropy_error": error,
        "perplexity": benchmark["perplexity"],
    }
