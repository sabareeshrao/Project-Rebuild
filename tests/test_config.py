import pytest
from rebuild_gpt.config import GPTConfig


def test_config_records_depth_width_and_per_head_size():
    config = GPTConfig(vocab_size=40, block_size=16, n_embd=12, n_head=3, n_layer=4)
    assert config.head_size == 4
    assert config.dropout == 0.0


@pytest.mark.parametrize("field", ["vocab_size", "block_size", "n_embd", "n_head", "n_layer"])
def test_config_rejects_nonpositive_dimensions(field):
    settings = dict(vocab_size=10, block_size=4, n_embd=8, n_head=2, n_layer=2)
    settings[field] = 0
    with pytest.raises(ValueError, match=field):
        GPTConfig(**settings)


def test_config_rejects_nondivisible_heads_and_invalid_dropout():
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(10, 4, 7, 2, 2)
    for rate in (-0.1, 1.01, float("nan")):
        with pytest.raises(ValueError, match="dropout"):
            GPTConfig(10, 4, 8, 2, 2, dropout=rate)


def test_config_is_immutable():
    from dataclasses import FrozenInstanceError
    config = GPTConfig(10, 4, 8, 2, 2)
    with pytest.raises(FrozenInstanceError):
        config.n_layer = 4


def test_tying_flag_must_be_boolean():
    assert GPTConfig(10, 4, 8, 2, 2).tie_embeddings is True
    with pytest.raises(ValueError, match="tie_embeddings"):
        GPTConfig(10, 4, 8, 2, 2, tie_embeddings=1)


def test_attention_implementation_is_validated():
    assert GPTConfig(10, 4, 8, 2, 2).attention_impl == "educational"
    assert GPTConfig(10, 4, 8, 2, 2, attention_impl="fused").attention_impl == "fused"
    with pytest.raises(ValueError, match="attention_impl"):
        GPTConfig(10, 4, 8, 2, 2, attention_impl="unknown")
