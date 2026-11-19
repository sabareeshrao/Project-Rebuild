"""Caching views must not change sampling distribution or RNG continuity."""

import pytest
import torch

from rebuild_gpt.data import sample_batch
from rebuild_gpt.window_sampler import CachedWindowSampler


@pytest.mark.parametrize("block_size,batch_size", [(1, 1), (3, 6), (7, 4)])
def test_cache_matches_reference_across_multiple_samples(block_size, batch_size):
    tokens = torch.arange(113, dtype=torch.long) % 17
    sampler = CachedWindowSampler(tokens, block_size=block_size)
    rng_reference = torch.Generator().manual_seed(157)
    rng_cached = torch.Generator().manual_seed(157)
    for _ in range(5):
        expected = sample_batch(
            tokens, batch_size=batch_size, block_size=block_size,
            generator=rng_reference,
        )
        actual = sampler.sample(batch_size=batch_size, generator=rng_cached)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        assert torch.equal(rng_reference.get_state(), rng_cached.get_state())


def test_constructor_does_not_consume_global_rng():
    state = torch.get_rng_state().clone()
    CachedWindowSampler(torch.arange(64), block_size=5)
    assert torch.equal(state, torch.get_rng_state())


def test_cached_windows_share_original_token_storage():
    tokens = torch.arange(50)
    sampler = CachedWindowSampler(tokens, block_size=8)
    assert sampler.windows.shape == (42, 9)
    assert sampler.windows.untyped_storage().data_ptr() == tokens.untyped_storage().data_ptr()
    assert sampler.windows.untyped_storage().nbytes() == tokens.untyped_storage().nbytes()


@pytest.mark.parametrize("batch_size", [0, -1, True, "3"])
def test_invalid_batch_size_is_rejected(batch_size):
    sampler = CachedWindowSampler(torch.arange(12), block_size=3)
    with pytest.raises(ValueError, match="batch_size"):
        sampler.sample(batch_size=batch_size)


def test_invalid_corpus_and_context_are_rejected():
    with pytest.raises(ValueError, match="flat torch.long"):
        CachedWindowSampler(torch.ones(2, 3, dtype=torch.long), block_size=2)
    with pytest.raises(ValueError, match="block_size"):
        CachedWindowSampler(torch.arange(3), block_size=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_cache_and_reference_use_identical_gpu_generator():
    tokens = (torch.arange(100, device="cuda") % 13).long()
    a = torch.Generator(device="cuda").manual_seed(157)
    b = torch.Generator(device="cuda").manual_seed(157)
    sampler = CachedWindowSampler(tokens, block_size=4)
    expected = sample_batch(tokens, batch_size=5, block_size=4, generator=a)
    actual = sampler.sample(batch_size=5, generator=b)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
