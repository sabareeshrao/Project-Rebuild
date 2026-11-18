import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict, load_nanogpt_state_dict, upstream_key


def model():
    return GPTLanguageModel(GPTConfig(9, 5, 8, 2, 2, attention_impl="fused"))


def test_key_translation_matches_upstream_module_layout():
    assert upstream_key("tokens.lookup.weight") == "transformer.wte.weight"
    assert upstream_key("blocks.blocks.1.attention.qkv.c_attn.weight") == "transformer.h.1.attn.c_attn.weight"
    assert upstream_key("blocks.blocks.1.mlp.expand.bias") == "transformer.h.1.mlp.c_fc.bias"
    assert upstream_key("final_norm.bias") == "transformer.ln_f.bias"
    assert upstream_key("head.linear.weight") == "lm_head.weight"
    with pytest.raises(ValueError, match="Unmapped"):
        upstream_key("unknown.weight")


def test_export_import_roundtrip_preserves_weights_and_outputs():
    torch.manual_seed(3)
    original = model().eval()
    exported = export_nanogpt_state_dict(original)
    assert "transformer.h.0.attn.c_attn.weight" in exported
    assert exported["transformer.h.0.attn.c_attn.weight"].shape == (24, 8)
    restored = model().eval()
    load_nanogpt_state_dict(restored, exported)
    x = torch.tensor([[0, 2, 3, 4]])
    with torch.no_grad():
        assert torch.equal(original(x), restored(x))
    assert restored.tokens.lookup.weight is restored.head.linear.weight


def test_import_rejects_missing_and_bad_shapes():
    checkpoint = export_nanogpt_state_dict(model())
    absent = dict(checkpoint)
    absent.pop("transformer.wpe.weight")
    with pytest.raises(ValueError, match="key mismatch"):
        load_nanogpt_state_dict(model(), absent)
    malformed = dict(checkpoint)
    malformed["transformer.wpe.weight"] = torch.zeros(1, 8)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_nanogpt_state_dict(model(), malformed)


def test_import_rejects_conflicting_tied_weights():
    checkpoint = export_nanogpt_state_dict(model())
    checkpoint["lm_head.weight"] = checkpoint["lm_head.weight"] + 1
    with pytest.raises(ValueError, match="Tied embedding"):
        load_nanogpt_state_dict(model(), checkpoint)


def test_nonfused_or_untied_paths_are_not_exportable():
    with pytest.raises(ValueError, match="fused"):
        export_nanogpt_state_dict(GPTLanguageModel(GPTConfig(9, 5, 8, 2, 1)))
    with pytest.raises(ValueError, match="fused"):
        export_nanogpt_state_dict(GPTLanguageModel(GPTConfig(9, 5, 8, 2, 1, tie_embeddings=False, attention_impl="fused")))
