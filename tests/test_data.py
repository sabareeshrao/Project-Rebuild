import pytest
import torch

from rebuild_gpt.data import load_corpus


def test_load_corpus_encodes_and_restores_text(tmp_path):
    path = tmp_path / "story.txt"
    path.write_text("hello\nhi", encoding="utf-8")
    stream, tokenizer = load_corpus(path)
    assert stream.ndim == 1
    assert stream.dtype == torch.long
    assert tokenizer.decode(stream.tolist()) == "hello\nhi"


def test_empty_corpus_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_corpus(path)
