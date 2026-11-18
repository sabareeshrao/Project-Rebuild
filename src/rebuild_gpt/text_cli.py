"""Command-line text generation from a self-contained GPT checkpoint."""

import argparse

import torch

from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.text_generation import generate_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Saved GPT + character vocab .pt file")
    parser.add_argument("--prompt", required=True, help="Non-empty prompt using saved characters")
    parser.add_argument("--new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    model, tokenizer = load_text_checkpoint(args.checkpoint)
    generator = torch.Generator().manual_seed(args.seed)
    try:
        result = generate_text(
            model, tokenizer, args.prompt, args.new_tokens,
            temperature=args.temperature, top_k=args.top_k,
            generator=generator,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
