"""Equal-looking embedding weights must really be one optimizer parameter."""

import pytest
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.embedding_tie_check import check_embedding_tying
from rebuild_gpt.gpt import GPTLanguageModel


def test_default_gpt_has_one_shared_embedding_parameter():
    model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    report = check_embedding_tying(model)
    assert report["tied"] is True
    assert report["vocab_size"] == 11
    assert report["embedding_elements"] == 88
    assert report["unique_parameter_tensors"] == len(list(model.parameters()))


def test_cloned_equal_values_are_not_real_weight_tying():
    model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    model.head.linear.weight = nn.Parameter(
        model.tokens.lookup.weight.detach().clone()
    )
    assert model.head.linear.weight is not model.tokens.lookup.weight
    with pytest.raises(AssertionError, match="share one Parameter"):
        check_embedding_tying(model)


def test_intentionally_untied_model_is_out_of_scope():
    model = GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, tie_embeddings=False, attention_impl="fused",
    ))
    with pytest.raises(ValueError, match="disabled"):
        check_embedding_tying(model)
