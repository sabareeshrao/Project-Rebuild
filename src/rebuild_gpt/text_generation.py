"""Generate text with a character tokenizer and a trained GPT model."""

import torch

from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.tokenizer import CharTokenizer


def generate_text(
    model: GPTLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Encode a text prompt, sample token IDs, and decode the full result.

    Temporarily select evaluation mode to disable dropout, then restore
    the model's prior train/eval mode even if sampling raises an error.
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocab_size must match tokenizer vocab_size")
    token_ids = tokenizer.encode(prompt)  # Unseen characters should fail loudly.
    device = next(model.parameters()).device
    encoded = torch.tensor([token_ids], dtype=torch.long, device=device)

    was_training = model.training
    try:
        model.eval()
        generated = model.generate(
            encoded, max_new_tokens, temperature=temperature,
            top_k=top_k, generator=generator,
        )
    finally:
        model.train(was_training)
    return tokenizer.decode(generated[0].tolist())
