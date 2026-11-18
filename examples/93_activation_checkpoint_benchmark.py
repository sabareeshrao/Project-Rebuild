"""Compare the training compute / memory tradeoff from block checkpointing.

Both modes use identical model initialization and token batches. Timing
is real wall time; CUDA allocator peaks require actual CUDA hardware.
"""

import argparse
import gc

import torch

from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.cuda_memory import measure_cuda_peak
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.memory import mebibytes
from rebuild_gpt.precision import make_grad_scaler, resolve_precision
from rebuild_gpt.training import train_step


def measure_variant(
    *,
    enabled: bool,
    config: GPTConfig,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    precision: str,
    iterations: int,
    warmup: int,
) -> tuple[int, float, float | None]:
    """Build and train a model with fresh, reproducible starting weights."""
    torch.manual_seed(125)
    model = GPTLanguageModel(config).to(device)
    model.blocks.set_activation_checkpointing(enabled)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = make_grad_scaler(device, precision)

    def step():
        return train_step(
            model, optimizer, tokens, targets,
            precision=precision, scaler=scaler,
        )

    timing = time_operation(
        step, device=device, iterations=iterations, warmup=warmup
    )
    peak_mib = None
    if device.type == "cuda":
        # AdamW moment tensors are already initialized by timed updates;
        # we measure the incremental allocation above that baseline.
        memory = measure_cuda_peak(step, device=device)
        peak_mib = mebibytes(memory.additional_peak_allocated_bytes)
    parameter_count = sum(p.numel() for p in model.parameters())
    del model, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return parameter_count, timing.median_ms, peak_mib


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--attention-impl", choices=("fused", "sdpa"), default="sdpa")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    args = parser.parse_args(argv)
    if (args.iterations < 1 or args.warmup < 0 or args.batch_size < 1
            or args.block_size < 1 or args.n_layer < 1):
        parser.error("iterations, batch-size, block-size and n-layer must be positive; warmup nonnegative")
    try:
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    config = GPTConfig(
        vocab_size=32, block_size=args.block_size, n_embd=32,
        n_head=4, n_layer=args.n_layer,
        attention_impl=args.attention_impl,
    )
    torch.manual_seed(126)
    tokens = torch.randint(
        config.vocab_size, (args.batch_size, args.block_size),
        device=device,
    )
    targets = torch.randint(
        config.vocab_size, (args.batch_size, args.block_size),
        device=device,
    )

    print(f"device={device} precision={args.precision} "
          f"backend={args.attention_impl} n_layer={args.n_layer}")
    measurements = {}
    for enabled in (False, True):
        label = "on" if enabled else "off"
        count, median_ms, peak = measure_variant(
            enabled=enabled, config=config, tokens=tokens, targets=targets,
            device=device, precision=args.precision,
            iterations=args.iterations, warmup=args.warmup,
        )
        measurements[label] = median_ms
        print(f"checkpointing={label} parameters={count} "
              f"training median={median_ms:.3f} ms")
        if peak is None:
            print("  additional CUDA peak: unavailable (CPU)")
        else:
            print(f"  additional CUDA peak allocated: {peak:.4f} MiB")
    print(f"checkpointed / regular latency ratio: "
          f"{measurements['on'] / measurements['off']:.3f}x")
    print("The CUDA peak is an allocator delta, not total VRAM consumption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
