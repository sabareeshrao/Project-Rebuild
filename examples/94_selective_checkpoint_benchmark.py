"""Benchmark no, selective, and all-block activation checkpointing.

Results are hardware dependent. CPU benchmarks report latency only;
real CUDA memory allocator peaks require a compatible GPU.
"""

import argparse
import gc

import torch

from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.checkpoint_policy import checkpointed_indices
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.cuda_memory import measure_cuda_peak
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.memory import mebibytes
from rebuild_gpt.precision import make_grad_scaler, resolve_precision
from rebuild_gpt.training import train_step


def measure_mode(
    *, interval: int | None, config: GPTConfig,
    inputs: torch.Tensor, targets: torch.Tensor,
    device: torch.device, precision: str,
    iterations: int, warmup: int,
) -> tuple[tuple[int, ...], int, float, float | None]:
    """Train an identical GPT under one recomputation policy."""
    torch.manual_seed(130)
    model = GPTLanguageModel(config).to(device)
    model.blocks.set_checkpoint_interval(interval)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = make_grad_scaler(device, precision)

    def step() -> float:
        return train_step(
            model, optimizer, inputs, targets,
            precision=precision, scaler=scaler,
        )

    median_ms = time_operation(
        step, device=device, iterations=iterations, warmup=warmup,
    ).median_ms
    additional_peak_mib = None
    if device.type == "cuda":
        # The timed optimizer steps have already initialized AdamW moments.
        # Measure the next step above the current allocator baseline.
        additional_peak_mib = mebibytes(
            measure_cuda_peak(step, device=device).additional_peak_allocated_bytes
        )
    result = (
        model.blocks.checkpointed_blocks,
        sum(p.numel() for p in model.parameters()),
        median_ms, additional_peak_mib,
    )
    del model, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--attention-impl", choices=("fused", "sdpa"), default="sdpa")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--checkpoint-every-n", type=int, default=2)
    args = parser.parse_args(argv)
    if (args.iterations < 1 or args.warmup < 0
            or args.batch_size < 1 or args.block_size < 1
            or args.n_layer < 1):
        parser.error("iterations, batch-size, block-size and n-layer must be positive; warmup nonnegative")
    try:
        checkpointed_indices(args.n_layer, args.checkpoint_every_n)
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    config = GPTConfig(
        vocab_size=32, block_size=args.block_size, n_embd=16,
        n_head=4, n_layer=args.n_layer, attention_impl=args.attention_impl,
    )
    torch.manual_seed(131)
    inputs = torch.randint(
        0, config.vocab_size, (args.batch_size, args.block_size), device=device,
    )
    targets = torch.randint(
        0, config.vocab_size, (args.batch_size, args.block_size), device=device,
    )
    print(f"device={device} precision={args.precision} "
          f"backend={args.attention_impl} n_layer={args.n_layer}")
    measurements = {}
    parameter_counts = set()
    modes = [
        ("off", None),
        (f"every_{args.checkpoint_every_n}", args.checkpoint_every_n),
        ("all", 1),
    ]
    for name, interval in modes:
        selected, count, latency, peak = measure_mode(
            interval=interval, config=config, inputs=inputs, targets=targets,
            device=device, precision=args.precision,
            iterations=args.iterations, warmup=args.warmup,
        )
        measurements[name] = latency
        parameter_counts.add(count)
        print(f"mode={name} blocks={selected} parameters={count} "
              f"train median={latency:.3f} ms")
        if peak is None:
            print("  additional CUDA peak: unavailable (CPU)")
        else:
            print(f"  additional CUDA peak allocated: {peak:.4f} MiB")
    assert len(parameter_counts) == 1, "checkpointing should not change parameters"
    print(f"selective / off latency ratio: "
          f"{measurements[modes[1][0]] / measurements['off']:.3f}x")
    print(f"all / off latency ratio: {measurements['all'] / measurements['off']:.3f}x")
    print("CUDA peaks are allocator deltas, not total GPU memory; timings are not guarantees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
