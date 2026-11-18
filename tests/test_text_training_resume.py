"""A resumed text-training run must match one uninterrupted run."""

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def test_resuming_reproduces_uninterrupted_weights_and_best_history(tmp_path):
    corpus = tmp_path / "repeat.txt"
    corpus.write_text("ab" * 100, encoding="utf-8")
    state = tmp_path / "state.pt"
    options = dict(
        block_size=3, n_embd=8, n_head=2, n_layer=1,
        batch_size=3, seed=13, learning_rate=0.02,
        eval_interval=2, eval_batches=2, restore_best=True,
        gradient_accumulation_steps=2,
    )
    early_history = []
    train_char_gpt(
        corpus, steps=2, history=early_history,
        training_state_path=state, **options,
    )
    resumed_history = []
    resumed_model, resumed_tok, resumed_metrics = train_char_gpt(
        corpus, steps=4, history=resumed_history,
        resume_from=state, **options,
    )
    complete_history = []
    complete_model, complete_tok, complete_metrics = train_char_gpt(
        corpus, steps=4, history=complete_history, **options,
    )
    assert resumed_history == complete_history
    assert resumed_metrics == complete_metrics
    assert resumed_tok.vocab == complete_tok.vocab
    assert all(
        torch.equal(p, q)
        for p, q in zip(resumed_model.parameters(), complete_model.parameters())
    )


def test_resume_rejects_changed_corpus_or_options(tmp_path):
    corpus = tmp_path / "text.txt"
    corpus.write_text("ab" * 100, encoding="utf-8")
    state = tmp_path / "state.pt"
    kw = dict(block_size=3, n_embd=8, n_head=2, n_layer=1, steps=2)
    train_char_gpt(corpus, training_state_path=state, **kw)
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(
            corpus, resume_from=state, batch_size=4, steps=4,
            block_size=3, n_embd=8, n_head=2, n_layer=1,
        )
    with pytest.raises(ValueError, match="steps must exceed"):
        train_char_gpt(corpus, resume_from=state, **kw)
    corpus.write_text("ba" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="corpus_sha256"):
        train_char_gpt(corpus, resume_from=state, steps=4,
                       block_size=3, n_embd=8, n_head=2, n_layer=1)
