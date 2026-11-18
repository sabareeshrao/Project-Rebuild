import pytest

from rebuild_gpt.tokenizer import CharTokenizer


def test_char_vocabulary_is_sorted_and_deterministic():
    first = CharTokenizer("cab cab")
    second = CharTokenizer("abc ")
    assert first.vocab == (" ", "a", "b", "c")
    assert first.stoi == second.stoi
    assert first.vocab_size == 4


def test_empty_text_cannot_create_vocabulary():
    with pytest.raises(ValueError, match="empty"):
        CharTokenizer("")
