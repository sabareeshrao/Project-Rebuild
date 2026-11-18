"""Choose how GPT calculations use floating point; weights remain FP32."""

from contextlib import nullcontext

import torch


def resolve_precision(device: torch.device, precision: str = "fp32") -> torch.dtype | None:
    """Return autocast dtype, or None for ordinary FP32 operations.

    CPU supports BF16 autocast; FP16 is only offered on CUDA with scaling.
    CUDA BF16 requires hardware support rather than silently falling back.
    """
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("precision must be fp32, bf16, or fp16")
    if precision == "fp32":
        return None
    if device.type not in ("cpu", "cuda"):
        raise ValueError("mixed precision requires a CPU or CUDA device")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 mixed precision requires CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("mixed precision requested but CUDA is unavailable")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support bf16")
    return torch.bfloat16 if precision == "bf16" else torch.float16


def autocast_context(device: torch.device, precision: str = "fp32"):
    """Autocast the *forward and loss* only; optimizer weights stay FP32."""
    dtype = resolve_precision(device, precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)
