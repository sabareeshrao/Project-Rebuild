"""Lesson 292: run a fresh tiny-GPT final learning demonstration."""

import sys

from rebuild_gpt.release_review_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["--mode", "train", *sys.argv[1:]]))
