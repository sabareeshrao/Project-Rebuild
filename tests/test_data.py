import pytest
import torch

from rebuild_gpt.data import load_corpus, sample_batch, split_train_val


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


def test_sample_batch_returns_aligned_next_token_pairs():
    stream = torch.arange(20, dtype=torch.long)
    x, y = sample_batch(stream, batch_size=4, block_size=5, generator=torch.Generator().manual_seed(42))
    assert x.shape == (4, 5)
    assert y.shape == (4, 5)
    assert torch.equal(y, x + 1)


def test_sampling_with_same_seed_is_reproducible():
    stream = torch.arange(30, dtype=torch.long)
    a = sample_batch(stream, batch_size=6, block_size=4, generator=torch.Generator().manual_seed(7))
    b = sample_batch(stream, batch_size=6, block_size=4, generator=torch.Generator().manual_seed(7))
    assert all(torch.equal(left, right) for left, right in zip(a, b))


@pytest.mark.parametrize("batch_size,block_size", [(0, 3), (2, 0), (-1, 3)])
def test_sampling_rejects_nonpositive_sizes(batch_size, block_size):
    with pytest.raises(ValueError, match="positive"):
        sample_batch(torch.arange(10), batch_size=batch_size, block_size=block_size)


def test_sampling_rejects_context_larger_than_stream():
    with pytest.raises(ValueError, match="block_size"):
        sample_batch(torch.arange(4), batch_size=2, block_size=4)


def test_sampling_rejects_wrong_tensor_shape_or_dtype():
    with pytest.raises(ValueError, match="flat torch.long"):
        sample_batch(torch.zeros(5, 3), batch_size=2, block_size=2)
    with pytest.raises(ValueError, match="flat torch.long"):
        sample_batch(torch.arange(10, dtype=torch.float32), batch_size=2, block_size=2)
