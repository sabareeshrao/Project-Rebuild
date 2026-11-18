"""Compare estimated training storage and optional actual CUDA peak memory.

Usage: python examples/90_memory_comparison.py --device cpu
On CUDA, one warmup step materializes optimizer state before peak profiling.
Run variants separately to avoid overlapping live model allocations.
"""

import argparse
import gc

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.cuda_memory import measure_cuda_peak
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.memory import estimate_model_memory, mebibytes
from rebuild_gpt.precision import make_grad_scaler, resolve_precision
from rebuild_gpt.training import train_step


def benchmark_model(
    implementation: str, *,
    device: torch.device, precision: str, batch_size: int, block_size: int,
) -> tuple[int, float, float | None]:
    torch.manual_seed(116)
    config = GPTConfig(
        vocab_size=32, block_size=block_size,
        n_embd=32, n_head=4, n_layer=2,
        attention_impl=implementation,
    )
    model = GPTLanguageModel(config).to(device)
    storage = estimate_model_memory(model)
    inputs = torch.randint(0, 32, (batch_size, block_size), device=device)
    targets = torch.randint(0, 32, (batch_size, block_size), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = make_grad_scaler(device, precision)

    # Model and optimizer are alive during the memory measurement.
    def step():
        train_step(
            model, optimizer, inputs, targets,
            precision=precision, scaler=scaler,
        )

    # One untimed warmup builds optimizer moments and initializes kernels.
    step()
    peak_mib = None
    if device.type == "cuda":
        measurement = measure_cuda_peak(step, device=device)
        peak_mib = mebibytes(measurement.additional_peak_allocated_bytes)
    result = (
        storage.trainable_parameters,
        mebibytes(storage.estimated_training_bytes),
        peak_mib,
    )
    del model, optimizer, scaler, inputs, targets
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.block_size < 1:
        parser.error("batch-size and block-size must be positive")
    try:
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"device={device} precision={args.precision} batch={args.batch_size} context={args.block_size}")
    for implementation in ("fused", "sdpa"):
        params, estimate_mib, measured = benchmark_model(
            implementation, device=device, precision=args.precision,
            batch_size=args.batch_size, block_size=args.block_size,
        )
        print(f"{implementation}: trainable_parameters={params}")
        print(f"  estimated persistent training tensors: {estimate_mib:.4f} MiB")
        if measured is None:
            print("  measured additional CUDA peak: n/a (CPU only)")
        else:
            print(f"  measured additional CUDA peak allocated: {measured:.4f} MiB")
    print("Estimates exclude activations; CUDA delta excludes existing model/optimizer allocations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
