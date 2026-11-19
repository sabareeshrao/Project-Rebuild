"""Explicit host-memory staging for educational CPU-to-CUDA transfers."""

import torch


def stage_cpu_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally copy a CPU logical batch into page-locked host memory.

    Pin once per logical batch, not once per microbatch. Calling pin_memory()
    itself can copy data and adds CPU overhead. No CUDA APIs are touched
    when pin_memory=False, so the default remains CPU-only compatible.
    """
    if type(pin_memory) is not bool:
        raise ValueError("pin_memory must be boolean")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError("inputs and targets must be tensors")
    if not pin_memory:
        return inputs, targets
    if device.type != "cuda":
        raise ValueError("pin_memory requires a CUDA compute device")
    if inputs.device.type != "cpu" or targets.device.type != "cpu":
        raise ValueError("pin_memory requires CPU source tensors")
    if not torch.cuda.is_available():
        raise ValueError("pin_memory requires an available CUDA accelerator")
    return (
        inputs if inputs.is_pinned() else inputs.pin_memory(),
        targets if targets.is_pinned() else targets.pin_memory(),
    )


def transfer_microbatch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a chunk to its compute device; async copy requires pinned CPU.

    The transfer and subsequent model operation must use the same
    CUDA stream for correct ordering. This helper does not create a
    background stream or promise overlap with computation.
    """
    if type(non_blocking) is not bool:
        raise ValueError("non_blocking must be boolean")
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA torch.device")
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError("inputs and targets must be tensors")
    if non_blocking:
        if device.type != "cuda":
            raise ValueError("non_blocking transfer requires CUDA")
        if (
            inputs.device.type != "cpu" or targets.device.type != "cpu"
            or not inputs.is_pinned() or not targets.is_pinned()
        ):
            raise ValueError("non_blocking requires pinned CPU inputs and targets")
    return (
        inputs.to(device, non_blocking=non_blocking),
        targets.to(device, non_blocking=non_blocking),
    )
