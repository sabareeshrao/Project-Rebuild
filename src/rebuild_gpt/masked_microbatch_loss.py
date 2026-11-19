"""Prove masked microbatches require supervised-token-weighted losses."""

import torch
from torch.nn import functional as F

from rebuild_gpt.perplexity_reference import compare_uniform_perplexity


def check_masked_microbatch_equivalence(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    split_rows: int = 1,
    atol: float = 2e-6,
) -> dict:
    """Match full-batch loss and logit gradients after splitting batch rows.

    Each nonempty microbatch's mean CE must be weighted by its count
    of supervised positions. Unweighted mean-of-means is wrong when
    ignore_index masks different numbers of positions per microbatch.
    """
    baseline = compare_uniform_perplexity(logits, targets)
    if logits.size(0) < 2:
        raise ValueError("At least two batch rows are required")
    if type(split_rows) is not int or not 1 <= split_rows < logits.size(0):
        raise ValueError("split_rows must divide the batch into two nonempty parts")
    total = baseline["supervised_positions"]
    whole = logits.detach().clone().requires_grad_(True)
    split = logits.detach().clone().requires_grad_(True)
    full_ce = F.cross_entropy(
        whole.reshape(-1, whole.size(-1)), targets.reshape(-1),
        ignore_index=-1,
    )
    full_grad, = torch.autograd.grad(full_ce, whole)
    counts = []
    active_means = []
    weighted_loss = None
    for start, end in ((0, split_rows), (split_rows, targets.size(0))):
        micro_targets = targets[start:end]
        count = int((micro_targets != -1).sum())
        counts.append(count)
        if count == 0:
            continue
        part = split[start:end]
        part_ce = F.cross_entropy(
            part.reshape(-1, part.size(-1)),
            micro_targets.reshape(-1), ignore_index=-1,
        )
        active_means.append(part_ce)
        contribution = part_ce * (count / total)
        weighted_loss = contribution if weighted_loss is None else weighted_loss + contribution
    if weighted_loss is None:
        raise AssertionError("No supervised microbatch was found")
    weighted_grad, = torch.autograd.grad(weighted_loss, split)
    try:
        torch.testing.assert_close(full_ce, weighted_loss, atol=atol, rtol=0)
        torch.testing.assert_close(full_grad, weighted_grad, atol=atol, rtol=0)
    except AssertionError as exc:
        raise AssertionError("Token-weighted microbatch CE or gradients differ") from exc
    naive_unweighted = torch.stack(active_means).mean()
    return {
        "status": "matched",
        "microbatch_supervised_counts": counts,
        "total_supervised_positions": total,
        "full_cross_entropy": float(full_ce.detach()),
        "weighted_cross_entropy": float(weighted_loss.detach()),
        "loss_error": float((full_ce - weighted_loss).detach().abs()),
        "max_gradient_error": float((full_grad - weighted_grad).abs().max()),
        "naive_unweighted_mean_error": float(
            (naive_unweighted - full_ce).detach().abs()
        ),
    }
