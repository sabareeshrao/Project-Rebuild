import pytest
import torch

from rebuild_gpt.data import load_corpus, split_train_val


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


def test_split_keeps_order_and_nonoverlapping_parts():
    stream = torch.arange(10, dtype=torch.long)
    train, val = split_train_val(stream, val_fraction=0.2)
    assert train.tolist() == list(range(8))
    assert val.tolist() == [8, 9]


@pytest.mark.parametrize("fraction", [0, 1, -0.3, 1.5])
def test_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        split_train_val(torch.arange(20), fraction)


def test_split_rejects_short_or_nonflat_stream():
    with pytest.raises(ValueError, match="at least two"):
        split_train_val(torch.arange(5))
    with pytest.raises(ValueError, match="flat"):
        split_train_val(torch.zeros(2, 10))
