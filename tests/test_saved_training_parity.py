"""A saved-model training parity experiment must not rewrite a checkpoint."""

from hashlib import sha256
from unittest.mock import patch

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.saved_training_parity import compare_saved_capstone_adamw
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer
from rebuild_gpt.training_parity import AdamWStepParityResult


def test_saved_step_uses_real_seeded_tokens_and_preserves_checkpoint(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 55, encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)
    before_hash = sha256(checkpoint.read_bytes()).hexdigest()

    with (
        patch("rebuild_gpt.saved_training_parity.load_upstream_model",
              return_value=object()) as load,
        patch("rebuild_gpt.saved_training_parity.compare_one_adamw_step",
              return_value=AdamWStepParityResult(12, 1e-6, 13, 2e-6, 0.001)) as compare,
    ):
        report = compare_saved_capstone_adamw(
            checkpoint, corpus, "trusted/model.py",
            batch_size=3, seed=17, learning_rate=0.002,
        )
    assert report["batch_size"] == 3
    assert report["context_length"] == 4
    assert report["matched_gradients"] == 12
    assert report["max_updated_weight_error"] == 2e-6
    assert load.call_args.args[0] == "trusted/model.py"
    args, kwargs = compare.call_args
    assert args[2].shape == args[3].shape == (3, 4)
    assert torch.all(args[3][:, -1] == -1)
    assert kwargs == {"learning_rate": 0.002, "weight_decay": 0.01}
    assert sha256(checkpoint.read_bytes()).hexdigest() == before_hash
