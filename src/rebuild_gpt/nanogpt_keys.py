"""Translate our fused GPT tensor keys to the upstream nanoGPT naming scheme."""

from collections.abc import Mapping

import torch

from rebuild_gpt.gpt import GPTLanguageModel


_BLOCK_PARTS = {
    "ln1.": "ln_1.",
    "ln2.": "ln_2.",
    "attention.qkv.c_attn.": "attn.c_attn.",
    "attention.c_proj.": "attn.c_proj.",
    "mlp.expand.": "mlp.c_fc.",
    "mlp.project.": "mlp.c_proj.",
}


def upstream_key(local_name: str) -> str:
    """Map one named tensor; reject unmapped keys instead of guessing."""
    root = {
        "tokens.lookup.weight": "transformer.wte.weight",
        "positions.lookup.weight": "transformer.wpe.weight",
        "final_norm.weight": "transformer.ln_f.weight",
        "final_norm.bias": "transformer.ln_f.bias",
        "head.linear.weight": "lm_head.weight",
    }
    if local_name in root:
        return root[local_name]
    prefix = "blocks.blocks."
    if local_name.startswith(prefix):
        depth_and_part = local_name[len(prefix):].split(".", 1)
        if len(depth_and_part) == 2 and depth_and_part[0].isdigit():
            layer, suffix = depth_and_part
            for original, replacement in _BLOCK_PARTS.items():
                if suffix.startswith(original):
                    return f"transformer.h.{layer}.{replacement}{suffix[len(original):]}"
    raise ValueError(f"Unmapped local state key: {local_name}")


def _key_pairs(model: GPTLanguageModel) -> dict[str, str]:
    if model.config.attention_impl not in ("fused", "sdpa") or not model.config.tie_embeddings:
        raise ValueError("nanoGPT key mapping requires fused/SDPA attention and tied embeddings")
    return {name: upstream_key(name) for name in model.state_dict()}


def export_nanogpt_state_dict(model: GPTLanguageModel) -> dict[str, torch.Tensor]:
    """Export named tensors only; not an upstream training checkpoint."""
    return {
        remote: model.state_dict()[local].detach().clone()
        for local, remote in _key_pairs(model).items()
    }


def load_nanogpt_state_dict(
    model: GPTLanguageModel, tensors: Mapping[str, torch.Tensor]
) -> None:
    """Check names, shapes and tied weights before loading translated tensors."""
    pairs = _key_pairs(model)
    wanted = set(pairs.values())
    received = set(tensors)
    if wanted != received:
        raise ValueError(f"Checkpoint key mismatch: missing={sorted(wanted-received)}, extra={sorted(received-wanted)}")
    expected_shapes = {k: v.shape for k, v in model.state_dict().items()}
    for local, remote in pairs.items():
        if not isinstance(tensors[remote], torch.Tensor) or tensors[remote].shape != expected_shapes[local]:
            raise ValueError(f"Checkpoint shape mismatch for {remote}: expected {tuple(expected_shapes[local])}")
    if not torch.equal(tensors["transformer.wte.weight"], tensors["lm_head.weight"]):
        raise ValueError("Tied embedding and lm_head weights differ")
    model.load_state_dict({local: tensors[remote] for local, remote in pairs.items()}, strict=True)
