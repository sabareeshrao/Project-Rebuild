"""Export an opt-in operator timeline for a small GPT training run.

Open the resulting Chrome trace JSON in a compatible performance viewer.
Profiling records operations and allocations and adds its own overhead.
"""

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.precision import make_grad_scaler, resolve_precision
from rebuild_gpt.training import train_step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--attention-impl", choices=("fused", "sdpa"), default="sdpa")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--output", default="out/profiles/gpt_trace.json")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or args.block_size < 1:
        parser.error("steps, batch-size, and block-size must be positive")
    try:
        device = resolve_device(args.device)
        resolve_precision(device, args.precision)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(120)
    config = GPTConfig(
        vocab_size=32, block_size=args.block_size, n_embd=16,
        n_head=4, n_layer=1, attention_impl=args.attention_impl,
    )
    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = make_grad_scaler(device, args.precision)
    inputs = torch.randint(0, 32, (args.batch_size, args.block_size), device=device)
    targets = torch.randint(0, 32, (args.batch_size, args.block_size), device=device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as profiler:
        for _ in range(args.steps):
            train_step(
                model, optimizer, inputs, targets,
                precision=args.precision, scaler=scaler,
            )
    profiler.export_chrome_trace(str(output))
    print(f"Trace: {output}")
    print(f"device={device} precision={args.precision} backend={args.attention_impl}")
    print(profiler.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=8
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
