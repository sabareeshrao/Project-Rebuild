"""Run verification must detect changed inputs, weights and output text."""

import pytest

from rebuild_gpt.capstone import save_capstone_run, verify_capstone_run
from rebuild_gpt.capstone_spec import CapstoneSpec
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.tokenizer import CharTokenizer


@pytest.fixture
def saved_run(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("ab ab ab ab", encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(tokenizer.vocab_size, 4, 8, 2, 1)).eval()
    output = tmp_path / "run"
    save_capstone_run(
        model, tokenizer, {}, corpus, output,
        CapstoneSpec(new_tokens=3, prompt="a", seed=8),
    )
    return corpus, output


def test_verification_replays_generated_sample(saved_run):
    corpus, output = saved_run
    report = verify_capstone_run(corpus, output)
    assert report["generated_text"].startswith("a")


def test_verification_detects_changed_corpus(saved_run):
    corpus, output = saved_run
    corpus.write_text("ab ab changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus SHA-256"):
        verify_capstone_run(corpus, output)


def test_verification_detects_changed_checkpoint_before_loading(saved_run):
    corpus, output = saved_run
    (output / "model.pt").write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="Checkpoint SHA-256"):
        verify_capstone_run(corpus, output)


def test_verification_detects_changed_sample(saved_run):
    corpus, output = saved_run
    (output / "sample.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Saved sample"):
        verify_capstone_run(corpus, output)
