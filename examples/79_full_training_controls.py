"""Lesson 82: clip gradients, decay learning rate, score full splits, and save best."""

from pathlib import Path
from tempfile import TemporaryDirectory

from rebuild_gpt.train_cli import main as train_cli

with TemporaryDirectory() as folder:
    root = Path(folder)
    train_cli([
        "--corpus", "data/tiny_story.txt",
        "--steps", "30", "--eval-every", "10",
        "--validation-mode", "full",
        "--grad-clip", "1.0",
        "--warmup-steps", "3", "--lr-decay-steps", "25", "--min-lr", "0.002",
        "--output", str(root / "final.pt"),
        "--best-output", str(root / "best.pt"),
        "--history-file", str(root / "history.json"),
        "--restore-best",
    ])
