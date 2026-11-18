import json

import pytest

from rebuild_gpt.tokenizer import CharTokenizer
from rebuild_gpt.tokenizer_io import load_tokenizer, save_tokenizer


def test_saved_vocabulary_preserves_unicode_ids(tmp_path):
    tokenizer = CharTokenizer("café\n漢字 café")
    path = tmp_path / "nested" / "vocab.json"
    save_tokenizer(tokenizer, path)
    restored = load_tokenizer(path)
    assert restored.vocab == tokenizer.vocab
    assert restored.stoi == tokenizer.stoi
    assert restored.encode("漢字 café") == tokenizer.encode("漢字 café")
    assert restored.decode(tokenizer.encode("café\n")) == "café\n"


def test_from_vocab_reconstructs_ids_without_training_corpus():
    tok = CharTokenizer.from_vocab([" ", "a", "z"])
    assert tok.vocab == (" ", "a", "z")
    assert tok.encode("za") == [2, 1]


@pytest.mark.parametrize("vocab", [
    [], ["a", "a"], ["z", "a"], ["ab"], [1], "abc", None,
])
def test_bad_saved_vocabulary_is_rejected(vocab):
    with pytest.raises(ValueError, match="vocab"):
        CharTokenizer.from_vocab(vocab)


def test_unknown_tokenizer_version_is_rejected(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"format_version": 99, "vocab": ["a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_tokenizer(path)


def test_corrupt_vocabulary_is_rejected_when_loading(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"format_version": 1, "vocab": ["b", "a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted order"):
        load_tokenizer(path)
