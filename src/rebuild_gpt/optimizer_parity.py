"""Inspect why nanoGPT applies weight decay to some tensors, not others."""

from torch import nn


def parameter_rank_groups(model: nn.Module) -> dict[str, object]:
    """Report unique trainable parameters grouped by tensor rank.

    Tensor matrices (rank >= 2) receive weight decay in nanoGPT.
    Biases/LayerNorm vectors (rank < 2) do not. named_parameters()
    deduplicates the shared embedding/output matrix.
    """
    names: dict[str, list[str]] = {"decay": [], "no_decay": []}
    sizes = {"decay": 0, "no_decay": 0}
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        identity = id(param)
        if identity in seen:
            raise ValueError("Repeated parameter object encountered")
        seen.add(identity)
        group = "decay" if param.ndim >= 2 else "no_decay"
        names[group].append(name)
        sizes[group] += param.numel()
    if not seen:
        raise ValueError("Model contains no trainable parameters")
    return {
        "decay": sorted(names["decay"]),
        "no_decay": sorted(names["no_decay"]),
        "decay_elements": sizes["decay"],
        "no_decay_elements": sizes["no_decay"],
        "unique_parameter_tensors": len(seen),
        "total_elements": sum(sizes.values()),
    }
