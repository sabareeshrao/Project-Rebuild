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


def test_encoding_and_decoding_round_trip():
    tokenizer = CharTokenizer("hello world")
    tokens = tokenizer.encode("hello")
    assert len(tokens) == 5
    assert tokenizer.decode(tokens) == "hello"
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_unknown_character_raises_useful_error():
    tokenizer = CharTokenizer("hello")
    with pytest.raises(ValueError, match="Character"):
        tokenizer.encode("!")


def test_unknown_token_id_raises_useful_error():
    tokenizer = CharTokenizer("ab")
    with pytest.raises(ValueError, match="Token ID"):
        tokenizer.decode([20])
