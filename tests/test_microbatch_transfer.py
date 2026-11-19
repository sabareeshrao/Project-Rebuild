"""Separate synchronous copies from truly eligible nonblocking CUDA copies."""

import pytest
import torch

from rebuild_gpt.transfers import transfer_microbatch


def test_cpu_default_keeps_original_tensors():
    x = torch.arange(6).view(2, 3)
    y = x + 1
    a, b = transfer_microbatch(x, y, device=torch.device("cpu"))
    assert a is x and b is y


def test_nonblocking_cpu_does_not_silently_fall_back():
    x = torch.ones(3, 4)
    with pytest.raises(ValueError, match="requires CUDA"):
        transfer_microbatch(x, x, device=torch.device("cpu"), non_blocking=True)


def test_nonblocking_cuda_rejects_unpinned_sources():
    x = torch.ones(2, 3)
    with pytest.raises(ValueError, match="pinned CPU"):
        transfer_microbatch(x, x, device=torch.device("cuda"), non_blocking=True)


@pytest.mark.parametrize("value", [1, 0, None, "yes"])
def test_nonblocking_flag_must_be_boolean(value):
    x = torch.ones(2, 3)
    with pytest.raises(ValueError, match="boolean"):
        transfer_microbatch(x, x, device=torch.device("cpu"), non_blocking=value)


def test_invalid_device_and_input_are_rejected():
    with pytest.raises(ValueError, match="device"):
        transfer_microbatch(torch.ones(2), torch.ones(2), device=torch.device("meta"))
    with pytest.raises(ValueError, match="tensors"):
        transfer_microbatch([0, 1], torch.ones(2), device=torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_nonblocking_copy_preserves_ids_in_current_cuda_stream():
    from rebuild_gpt.transfers import stage_cpu_batch

    x = torch.arange(20, dtype=torch.long).view(4, 5)
    y = x + 1
    pinned_x, pinned_y = stage_cpu_batch(
        x, y, device=torch.device("cuda"), pin_memory=True,
    )
    copied_x, copied_y = transfer_microbatch(
        pinned_x, pinned_y, device=torch.device("cuda"), non_blocking=True,
    )
    torch.cuda.synchronize()
    assert copied_x.device.type == "cuda"
    assert torch.equal(copied_x.cpu(), x)
    assert torch.equal(copied_y.cpu(), y)
