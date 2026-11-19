"""Check reproducible GPT sampling without modifying the process RNG."""

import torch

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import _validate_token_batch


def check_seeded_generation(
    model: GPTLanguageModel,
    prompt: torch.Tensor,
    *,
    new_tokens: int = 6,
    seed: int = 37,
    temperature: float = 0.8,
    top_k: int | None = 3,
) -> dict:
    """Two private generators with the same seed should sample equal IDs.

    The model is temporarily switched to evaluation mode. Global CPU
    RNG state and the caller's original model mode are preserved.
    """
    _validate_token_batch(model, prompt)
    if next(model.parameters()).device.type != "cpu" or prompt.device.type != "cpu":
        raise ValueError("The seeded sampling check currently requires CPU tensors")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(new_tokens) is not int or new_tokens < 1:
        raise ValueError("new_tokens must be a positive integer")
    previous_mode = model.training
    previous_rng = torch.get_rng_state().clone()
    try:
        model.eval()
        with torch.no_grad():
            a = model.generate(
                prompt, new_tokens, temperature=temperature, top_k=top_k,
                generator=torch.Generator().manual_seed(seed),
            )
            b = model.generate(
                prompt, new_tokens, temperature=temperature, top_k=top_k,
                generator=torch.Generator().manual_seed(seed),
            )
        if a.shape != (prompt.size(0), prompt.size(1) + new_tokens):
            raise AssertionError("Generated token IDs have an unexpected shape")
        if not torch.equal(a[:, :prompt.size(1)], prompt):
            raise AssertionError("Generation changed the prompt prefix")
        if not torch.equal(a, b):
            raise AssertionError("Seeded generation differs between repeat runs")
        if not torch.equal(previous_rng, torch.get_rng_state()):
            raise AssertionError("Seeded generation modified the global CPU RNG")
        return {
            "seed": seed,
            "batch_size": prompt.size(0),
            "prompt_tokens": prompt.size(1),
            "new_tokens": new_tokens,
            "repeatable": True,
            "global_rng_preserved": True,
            "sampled_token_ids": a.tolist(),
        }
    finally:
        torch.set_rng_state(previous_rng)
        model.train(previous_mode)
