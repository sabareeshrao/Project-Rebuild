"""Compare equivalent manual fused and PyTorch SDPA GPT inference.

Results depend on machine, kernel, sequence length and precision. No fixed
speedup is promised. Run from repository root after editable installation.
"""

import argparse
from dataclasses import replace
from statistics import median
from time import perf_counter

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.precision import autocast_context, resolve_precision


def time_forward(
    model: GPTLanguageModel, tokens: torch.Tensor, *,
    precision: str, warmup: int, iterations: int,
) -> float:
    """Median forward latency in milliseconds, including launch sync on CUDA."""
    device = tokens.device
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model.eval()
    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast_context(device, precision):
                model(tokens)
        synchronize()
        for _ in range(iterations):
            synchronize()
            started = perf_counter()
            with autocast_context(device, precision):
                model(tokens)
            synchronize()
            samples.append((perf_counter() - started) * 1000.0)
    return median(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup < 0 or args.batch_size < 1 or args.block_size < 1:
        parser.error("iterations, batch-size, block-size must be positive; warmup nonnegative")
    try:
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(17)
    config = GPTConfig(
        vocab_size=64, block_size=args.block_size,
        n_embd=32, n_head=4, n_layer=2,
        attention_impl="fused",
    )
    manual = GPTLanguageModel(config).to(device).eval()
    sdpa = GPTLanguageModel(replace(config, attention_impl="sdpa")).to(device).eval()
    sdpa.load_state_dict(manual.state_dict(), strict=True)
    tokens = torch.randint(0, config.vocab_size,
                           (args.batch_size, args.block_size), device=device)
    with torch.inference_mode(), autocast_context(device, args.precision):
        maximum_difference = (manual(tokens) - sdpa(tokens)).abs().max().item()

    reference_ms = time_forward(
        manual, tokens, precision=args.precision,
        warmup=args.warmup, iterations=args.iterations,
    )
    sdpa_ms = time_forward(
        sdpa, tokens, precision=args.precision,
        warmup=args.warmup, iterations=args.iterations,
    )
    print(f"device={device} precision={args.precision} batch={args.batch_size} "
          f"context={args.block_size}")
    print(f"max absolute logit difference: {maximum_difference:.7g}")
    print(f"manual median: {reference_ms:.3f} ms")
    print(f"sdpa median:   {sdpa_ms:.3f} ms")
    print(f"manual / sdpa latency ratio: {reference_ms / sdpa_ms:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
