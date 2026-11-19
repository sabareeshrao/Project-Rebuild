"""Verify our capstone forward wrapper delegates to genuine parity checks."""

from unittest.mock import patch

from rebuild_gpt.capstone_parity import compare_capstone_forward
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer
from rebuild_gpt.upstream_parity import ParityResult


def test_saved_checkpoint_forward_parity_uses_actual_token_batches(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 50, encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)

    with (
        patch("rebuild_gpt.upstream_parity.load_upstream_model", return_value=object()) as load,
        patch(
            "rebuild_gpt.upstream_parity.compare_against_upstream",
            return_value=ParityResult(22, 0.000002, 0.000003),
        ) as compare,
    ):
        report = compare_capstone_forward(checkpoint, corpus, "reviewed_model.py", seed=11)

    assert report == {
        "batch_size": 2, "context_length": 4, "matched_tensors": 22,
        "max_logit_error": 0.000002, "loss_error": 0.000003,
    }
    load.assert_called_once()
    assert load.call_args.args[0] == "reviewed_model.py"
    _, _, x, y = compare.call_args.args
    assert x.shape == y.shape == (2, 4)
    assert (y[:, -1] == -1).all()
