"""Lesson 230: compare sampled next-token IDs from identical trained weights."""

import argparse

from rebuild_gpt.capstone_parity import compare_capstone_generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-model", required=True,
                        help="Path to a reviewed, trusted nanoGPT model.py")
    parser.add_argument("--checkpoint", default="out/capstone/model.pt")
    parser.add_argument("--prompt", default="a")
    parser.add_argument("--new-tokens", type=int, default=8)
    args = parser.parse_args()
    report = compare_capstone_generation(
        args.checkpoint, args.upstream_model,
        prompt=args.prompt, new_tokens=args.new_tokens,
    )
    for name, value in report.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
