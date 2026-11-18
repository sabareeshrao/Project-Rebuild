"""Compare our GPT against a separately obtained original nanoGPT model.py."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict


@dataclass(frozen=True)
class ParityResult:
    matched_tensors: int
    max_logit_error: float
    loss_error: float


def load_upstream_model(source_path: str | Path, config: GPTConfig) -> nn.Module:
    """Execute an explicitly supplied, trusted nanoGPT model.py, never download it."""
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Upstream model.py not found: {path}")
    spec = importlib.util.spec_from_file_location("_original_nanogpt_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import upstream model.py: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Required for the dataclass in upstream GPTConfig.
    spec.loader.exec_module(module)
    reference_config = module.GPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias,
    )
    return module.GPT(reference_config)


def synchronize_upstream_weights(
    ours: GPTLanguageModel, upstream: nn.Module
) -> int:
    """Strictly match all learnable tensors; permit only known causal-mask buffers."""
    exported = export_nanogpt_state_dict(ours)
    destination = upstream.state_dict()
    extra = set(destination) - set(exported)
    unexpected = [name for name in extra if not name.endswith(".attn.bias")]
    missing = set(exported) - set(destination)
    if missing or unexpected:
        raise ValueError(
            f"Reference key mismatch: missing={sorted(missing)}, extra={sorted(unexpected)}"
        )
    for key, tensor in exported.items():
        if destination[key].shape != tensor.shape:
            raise ValueError(
                f"Reference shape mismatch for {key}: "
                f"{tuple(destination[key].shape)} != {tuple(tensor.shape)}"
            )
    # Retain causal-mask buffers in older nanoGPT/PyTorch versions.
    updated = dict(destination)
    updated.update(exported)
    upstream.load_state_dict(updated, strict=True)
    return len(exported)


def compare_against_upstream(
    ours: GPTLanguageModel,
    upstream: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> ParityResult:
    """Compare full training logits, masked loss, and final-position inference."""
    if tokens.ndim != 2 or targets.shape != tokens.shape:
        raise ValueError("tokens and targets must both have shape [B,T]")
    matched = synchronize_upstream_weights(ours, upstream)
    ours_training, upstream_training = ours.training, upstream.training
    try:
        ours.eval()
        upstream.eval()
        with torch.no_grad():
            local_train_logits, local_loss = ours.forward_nanogpt(tokens, targets)
            ref_train_logits, ref_loss = upstream(tokens, targets)
            local_last, local_none = ours.forward_nanogpt(tokens)
            ref_last, ref_none = upstream(tokens)
        torch.testing.assert_close(local_train_logits, ref_train_logits, atol=atol, rtol=rtol)
        torch.testing.assert_close(local_last, ref_last, atol=atol, rtol=rtol)
        torch.testing.assert_close(local_loss, ref_loss, atol=atol, rtol=rtol)
        assert local_none is None and ref_none is None
        return ParityResult(
            matched_tensors=matched,
            max_logit_error=(local_train_logits - ref_train_logits).abs().max().item(),
            loss_error=abs((local_loss - ref_loss).item()),
        )
    finally:
        ours.train(ours_training)
        upstream.train(upstream_training)
