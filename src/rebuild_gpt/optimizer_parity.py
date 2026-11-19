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


def optimizer_decay_by_name(model: nn.Module, optimizer) -> dict[str, float]:
    """Read effective weight decay per UNIQUE trainable named parameter.

    Unlike comparing group order, comparing per-parameter values also
    works if optimizers list decay/no-decay groups in different orders.
    """
    name_by_id = {id(param): name for name, param in model.named_parameters()
                  if param.requires_grad}
    result: dict[str, float] = {}
    seen: set[int] = set()
    for group in optimizer.param_groups:
        decay = float(group["weight_decay"])
        for param in group["params"]:
            identity = id(param)
            if identity not in name_by_id:
                raise ValueError("Optimizer contains a parameter outside this model")
            if identity in seen:
                raise ValueError("Optimizer repeats a trainable parameter")
            seen.add(identity)
            result[name_by_id[identity]] = decay
    if seen != set(name_by_id):
        raise ValueError("Optimizer does not include all trainable parameters")
    return result


def compare_grouped_optimizer_layout(ours, upstream, local_optimizer, reference_optimizer,
                                     *, weight_decay: float) -> dict[str, int]:
    """Compare optimizer membership against *actual* upstream groups."""
    from rebuild_gpt.nanogpt_keys import upstream_key

    local = optimizer_decay_by_name(ours, local_optimizer)
    reference = optimizer_decay_by_name(upstream, reference_optimizer)
    renamed = {upstream_key(name): decay for name, decay in local.items()}
    if set(renamed) != set(reference):
        raise ValueError(
            "Optimizer parameter keys differ: missing="
            f"{sorted(set(renamed) - set(reference))} "
            f"extra={sorted(set(reference) - set(renamed))}"
        )
    for name, decay in renamed.items():
        if decay != reference[name]:
            raise ValueError(
                f"Optimizer weight decay mismatch for {name}: {decay} != {reference[name]}"
            )
    decayed = sum(decay == weight_decay and weight_decay != 0 for decay in renamed.values())
    excluded = sum(decay == 0.0 for decay in renamed.values())
    return {"matched_parameters": len(renamed), "decay_parameters": decayed,
            "no_decay_parameters": excluded}
