"""Opt-in comparison of sampled sequences with the original nanoGPT generator."""

from dataclasses import dataclass

import torch
from torch import nn

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.sampling import validate_temperature, validate_top_k
from rebuild_gpt.upstream_parity import synchronize_upstream_weights


@dataclass(frozen=True)
class GenerationParityResult:
    matched_tensors: int
    new_tokens: int
    temperature: float
    top_k: int | None


def compare_generation_against_upstream(
    ours: GPTLanguageModel,
    upstream: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 123,
) -> GenerationParityResult:
    """Transfer matching weights, then compare CPU token sequences exactly.

    Seed each run identically and restore caller RNG and train/eval modes.
    This comparison is intended for a trusted, pinned upstream model.
    """
    if prompt.device.type != "cpu":
        raise ValueError("Generation parity currently supports CPU prompts only")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    temperature = validate_temperature(temperature)
    top_k = validate_top_k(top_k)
    matched = synchronize_upstream_weights(ours, upstream)
    prior_rng = torch.get_rng_state().clone()
    ours_training, upstream_training = ours.training, upstream.training
    try:
        ours.eval()
        upstream.eval()
        torch.manual_seed(seed)
        actual = ours.generate(
            prompt, max_new_tokens, temperature=temperature, top_k=top_k
        )
        torch.manual_seed(seed)
        expected = upstream.generate(
            prompt, max_new_tokens, temperature=temperature, top_k=top_k
        )
        if not torch.equal(actual, expected):
            raise AssertionError(
                f"Generation tokens differ: ours={actual.tolist()}, upstream={expected.tolist()}"
            )
    finally:
        torch.set_rng_state(prior_rng)
        ours.train(ours_training)
        upstream.train(upstream_training)
    return GenerationParityResult(matched, max_new_tokens, temperature, top_k)
