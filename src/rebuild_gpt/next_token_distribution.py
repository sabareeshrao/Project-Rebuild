"""Inspect temperature, top-k support and entropy of GPT next-token sampling."""

import math

import torch

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.sampling import filter_top_k, scale_logits


def inspect_next_token_distribution(
    model: GPTLanguageModel,
    prompt: torch.Tensor,
    *,
    temperature: float = 0.8,
    top_k: int | None = 3,
) -> dict:
    """Return the exact sampling distribution used after context cropping.

    No token is actually sampled. Eval mode ensures dropout does not
    change the logits; the caller's model mode and CPU RNG are restored.
    """
    if (not isinstance(prompt, torch.Tensor)
            or prompt.ndim != 2 or prompt.dtype != torch.long
            or prompt.size(0) != 1 or prompt.size(1) < 1):
        raise ValueError("Expected one nonempty torch.long prompt [1, time]")
    if not bool(((prompt >= 0) & (prompt < model.config.vocab_size)).all()):
        raise ValueError("Prompt contains an out-of-vocabulary token")
    if prompt.device.type != "cpu" or next(model.parameters()).device.type != "cpu":
        raise ValueError("Sampling distribution inspection currently requires CPU")
    previous_mode = model.training
    previous_rng = torch.get_rng_state().clone()
    try:
        model.eval()
        with torch.no_grad():
            context = prompt[:, -model.config.block_size:]
            logits, _ = model.forward_nanogpt(context)
            candidate = filter_top_k(
                scale_logits(logits[:, -1, :], temperature), top_k,
            )
            probabilities = torch.softmax(candidate, dim=-1).squeeze(0)
        if not bool(torch.isfinite(probabilities).all()):
            raise AssertionError("Sampling distribution contains non-finite probabilities")
        if not math.isclose(float(probabilities.sum()), 1.0,
                            rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError("Sampling distribution is not normalized")
        support = (probabilities > 0).nonzero().flatten().tolist()
        if not support:
            raise AssertionError("No next-token candidates remain")
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * positive.log()).sum())
        return {
            "prompt_length": prompt.size(1),
            "context_length": context.size(1),
            "temperature": float(temperature),
            "top_k": top_k,
            "candidate_count": len(support),
            "allowed_ids": support,
            "peak_id": int(probabilities.argmax()),
            "entropy_nats": entropy,
            "probabilities": probabilities.tolist(),
        }
    finally:
        torch.set_rng_state(previous_rng)
        model.train(previous_mode)
