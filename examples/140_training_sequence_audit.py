"""Lesson 247: SHA-pinned optimizer parity on a varying training-batch plan."""

import sys

from rebuild_gpt.grouped_training_audit_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["--batch-mode", "train_sequence", *sys.argv[1:]]))
