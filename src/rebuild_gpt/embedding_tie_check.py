"""Inspect the weight tying shared by nanoGPT token embeddings and output head."""

import torch

from rebuild_gpt.gpt import GPTLanguageModel


def check_embedding_tying(model: GPTLanguageModel) -> dict[str, int | bool]:
    """Require one shared Parameter, not just two equal weight matrices.

    An equality test alone is insufficient: untied clones can begin
    equal and diverge after a training update.
    """
    if not model.config.tie_embeddings:
        raise ValueError("Embedding tying is disabled by model configuration")
    token_weight = model.tokens.lookup.weight
    output_weight = model.head.linear.weight
    if token_weight is not output_weight:
        raise AssertionError("Token embedding and output head do not share one Parameter")
    if not torch.equal(token_weight.detach(), output_weight.detach()):
        raise AssertionError("Tied weights unexpectedly have different values")
    ids = [id(param) for param in model.parameters()]
    if len(ids) != len(set(ids)):
        raise AssertionError("A tied Parameter would be updated more than once")
    named = dict(model.named_parameters())
    if "tokens.lookup.weight" not in named:
        raise AssertionError("Shared embedding weight is missing from named parameters")
    if any(param is output_weight and name != "tokens.lookup.weight"
           for name, param in named.items()):
        raise AssertionError("Tied embedding appears more than once in named parameters")
    return {
        "tied": True,
        "embedding_elements": token_weight.numel(),
        "unique_parameter_tensors": len(ids),
        "vocab_size": token_weight.shape[0],
    }
