import pytest

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict


@pytest.mark.parametrize("bias", [True, False])
def test_exact_upstream_style_keys_for_one_fused_block(bias):
    model = GPTLanguageModel(
        GPTConfig(9, 5, 8, 2, 1, attention_impl="fused", bias=bias)
    )
    tensors = export_nanogpt_state_dict(model)
    expected = {
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "lm_head.weight",
        "transformer.ln_f.weight",
    }
    layer = "transformer.h.0."
    for stem in ("ln_1", "ln_2"):
        expected.add(layer + stem + ".weight")
    for stem in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
        expected.add(layer + stem + ".weight")
    if bias:
        expected.add("transformer.ln_f.bias")
        for stem in ("ln_1", "ln_2", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            expected.add(layer + stem + ".bias")
    assert set(tensors) == expected
    assert len(tensors) == (17 if bias else 10)
