"""Cross-check GPT context cropping against an independent sampling loop."""

import torch

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.sampling import filter_top_k, scale_logits


def check_generation_context(
    model: GPTLanguageModel,
    prompt: torch.Tensor,
    *,
    new_tokens: int = 4,
    seed: int = 37,
    temperature: float = 0.8,
    top_k: int | None = 3,
) -> dict:
    """Compare generate() with manual next-token sampling using the LAST block.

    Intentionally starts from a prompt longer than block_size so the
    cropping behavior is genuinely exercised, not only assumed.
    """
    if (not isinstance(prompt, torch.Tensor) or prompt.ndim != 2
            or prompt.dtype != torch.long or prompt.size(0) < 1
            or prompt.size(1) <= model.config.block_size):
        raise ValueError("Prompt must exceed block_size with shape [batch, time]")
    if (prompt.device.type != "cpu"
            or next(model.parameters()).device.type != "cpu"):
        raise ValueError("Generation context check requires CPU")
    if not bool(((prompt >= 0) & (prompt < model.config.vocab_size)).all()):
        raise ValueError("Prompt contains a token ID outside vocabulary")
    if type(new_tokens) is not int or new_tokens < 1:
        raise ValueError("new_tokens must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    prior_mode = model.training
    prior_rng = torch.get_rng_state().clone()
    try:
        model.eval()
        with torch.no_grad():
            actual = model.generate(
                prompt, new_tokens, temperature=temperature, top_k=top_k,
                generator=torch.Generator().manual_seed(seed),
            )
            expected = prompt.clone()
            rng = torch.Generator().manual_seed(seed)
            for _ in range(new_tokens):
                context = expected[:, -model.config.block_size:]
                logits, _ = model.forward_nanogpt(context)
                scaled = scale_logits(logits[:, -1, :], temperature)
                sampled = torch.multinomial(
                    torch.softmax(filter_top_k(scaled, top_k), dim=-1),
                    1, generator=rng,
                )
                expected = torch.cat((expected, sampled), dim=1)
        if actual.shape != expected.shape:
            raise AssertionError("Generation context produced an incorrect output shape")
        if not torch.equal(actual, expected):
            raise AssertionError("Generation differs from independent cropped-context sampling")
        if not torch.equal(actual[:, :prompt.size(1)], prompt):
            raise AssertionError("Generation changed the original long prompt")
        if not torch.equal(prior_rng, torch.get_rng_state()):
            raise AssertionError("Generation consumed global CPU RNG state")
        return {
            "status": "matched",
            "prompt_tokens": prompt.size(1),
            "block_size": model.config.block_size,
            "sampled_tokens": new_tokens,
            "batch_size": prompt.size(0),
            "generated_ids": actual.tolist(),
        }
    finally:
        torch.set_rng_state(prior_rng)
        model.train(prior_mode)
