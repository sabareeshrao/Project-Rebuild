"""Compare vectorized and Python-loop next-token window gathering on CPU."""

import argparse

import torch

from rebuild_gpt.benchmarks import time_operation
from rebuild_gpt.data import gather_next_token_pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.block_size < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("batch, block and iterations must be positive; warmup nonnegative")

    tokens = torch.arange(65536, dtype=torch.long) % 32
    starts = torch.randint(
        0, len(tokens) - args.block_size,
        (args.batch_size,), generator=torch.Generator().manual_seed(152),
    )

    def vectorized():
        return gather_next_token_pairs(tokens, starts, block_size=args.block_size)

    def loop():
        # Python control flow adds overhead for each window.
        x = torch.stack([tokens[int(i):int(i) + args.block_size] for i in starts])
        y = torch.stack([tokens[int(i) + 1:int(i) + args.block_size + 1] for i in starts])
        return x, y

    x_fast, y_fast = vectorized()
    x_loop, y_loop = loop()
    torch.testing.assert_close(x_fast, x_loop, atol=0, rtol=0)
    torch.testing.assert_close(y_fast, y_loop, atol=0, rtol=0)
    results = {}
    for name, fn in (("vectorized", vectorized), ("python_loop", loop)):
        result = time_operation(
            fn, device=torch.device("cpu"),
            iterations=args.iterations, warmup=args.warmup,
        )
        results[name] = result.median_ms
        print(f"{name} median: {result.median_ms:.4f} ms")
    print(f"python_loop / vectorized latency: {results['python_loop'] / results['vectorized']:.3f}x")
    print("Both implementations gathered identical windows from identical fixed starts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
