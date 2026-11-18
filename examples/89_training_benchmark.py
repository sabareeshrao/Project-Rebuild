"""Compare complete training steps for manual attention and PyTorch SDPA.

Includes forward, backward, AdamW update, and CUDA synchronization.
Run from repository root after installing the package in editable mode.
"""

import argparse
from dataclasses import replace

import torch

from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.precision import autocast_context, make_grad_scaler, resolve_precision
from rebuild_gpt.training import train_step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup < 0 or args.batch_size < 1 or args.block_size < 1:
        parser.error("iterations, batch-size, block-size must be positive; warmup nonnegative")
    try:
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(113)
    config = GPTConfig(
        vocab_size=32, block_size=args.block_size,
        n_embd=32, n_head=4, n_layer=2, attention_impl="fused",
    )
    manual = GPTLanguageModel(config).to(device)
    sdpa = GPTLanguageModel(replace(config, attention_impl="sdpa")).to(device)
    sdpa.load_state_dict(manual.state_dict(), strict=True)

    ids = torch.randint(
        config.vocab_size, (args.batch_size, args.block_size), device=device
    )
    targets = torch.randint(
        config.vocab_size, (args.batch_size, args.block_size), device=device
    )
    manual.eval()
    sdpa.eval()
    with torch.no_grad(), autocast_context(device, args.precision):
        initial_difference = (manual(ids) - sdpa(ids)).abs().max().item()

    def run_backend(model: GPTLanguageModel):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = make_grad_scaler(device, args.precision)
        return time_operation(
            lambda: train_step(
                model, optimizer, ids, targets,
                precision=args.precision, scaler=scaler,
            ),
            device=device, iterations=args.iterations, warmup=args.warmup,
        )

    manual_timing = run_backend(manual)
    sdpa_timing = run_backend(sdpa)
    tokens_per_step = args.batch_size * args.block_size
    print(f"device={device} precision={args.precision}")
    print(f"tokens per optimizer step: {tokens_per_step}")
    print(f"initial max absolute logit difference: {initial_difference:.7g}")
    print(f"manual training-step median: {manual_timing.median_ms:.3f} ms")
    print(f"sdpa training-step median:   {sdpa_timing.median_ms:.3f} ms")
    print(f"manual / sdpa latency ratio: {manual_timing.median_ms / sdpa_timing.median_ms:.3f}x")
    print(f"manual tokens/sec: {tokens_per_step * 1000 / manual_timing.median_ms:.1f}")
    print(f"sdpa tokens/sec:   {tokens_per_step * 1000 / sdpa_timing.median_ms:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
