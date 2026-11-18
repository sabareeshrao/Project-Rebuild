import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def tiny_corpus(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("ab" * 90, encoding="utf-8")
    return path


def test_explicit_cpu_matches_original_cpu_default(tmp_path):
    corpus = tiny_corpus(tmp_path)
    options = dict(
        steps=2, block_size=3, n_embd=8, n_head=2, n_layer=1,
        batch_size=3, seed=24, eval_interval=1, eval_batches=1,
        gradient_accumulation_steps=2,
    )
    original, _, original_metrics = train_char_gpt(corpus, **options)
    explicit, _, explicit_metrics = train_char_gpt(corpus, device="cpu", **options)
    assert original_metrics == explicit_metrics
    assert next(explicit.parameters()).device.type == "cpu"
    assert all(
        torch.equal(x, y)
        for x, y in zip(original.parameters(), explicit.parameters())
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_model_trains_and_evaluates_cpu_corpus(tmp_path):
    corpus = tiny_corpus(tmp_path)
    model, tokenizer, metrics = train_char_gpt(
        corpus, steps=2, block_size=3, n_embd=8,
        n_head=2, n_layer=1, batch_size=3,
        eval_interval=1, eval_batches=1, seed=9,
        device="cuda", grouped_adamw=True,
        validation_mode="full",
    )
    assert next(model.parameters()).device.type == "cuda"
    assert tokenizer.vocab_size == 2
    assert metrics["train_loss_last"] > 0
    assert metrics["validation_after"] > 0


def test_unavailable_cuda_reports_clear_error(tmp_path, monkeypatch):
    corpus = tiny_corpus(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested"):
        train_char_gpt(corpus, steps=1, device="cuda")
