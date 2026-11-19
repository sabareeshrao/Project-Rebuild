"""Gradient snapshots for comparing our trained GPT to a trusted reference."""

import torch

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import upstream_key


def capture_upstream_named_gradients(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    """Clone one gradient per unique parameter, using nanoGPT-style names.

    Call loss.backward() first. PyTorch deduplicates tied embedding/head
    parameters in named_parameters(), so tied weights are counted once.
    """
    if model.config.attention_impl not in ("fused", "sdpa") or not model.config.tie_embeddings:
        raise ValueError("Gradient capture requires fused/SDPA attention and tied embeddings")
    result: dict[str, torch.Tensor] = {}
    for local_name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"Missing gradient for {local_name}; call backward() first")
        key = upstream_key(local_name)
        if key in result:
            raise ValueError(f"Duplicate mapped gradient name: {key}")
        result[key] = parameter.grad.detach().clone()
    return result
