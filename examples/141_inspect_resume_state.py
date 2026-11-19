"""Lesson 248: inspect persistent AdamW state and RNG information."""

import argparse

from rebuild_gpt.resume_inspection import inspect_resume_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default="out/resume_lab/partial.pt")
    args = parser.parse_args()
    for key, value in inspect_resume_checkpoint(args.checkpoint).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
