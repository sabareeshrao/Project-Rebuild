"""Lesson 87: run GPT training with AdamW grouping and accumulated microbatches."""

from pathlib import Path
from tempfile import TemporaryDirectory

from rebuild_gpt.text_cli import main as generate_text
from rebuild_gpt.train_cli import main as train_text


with TemporaryDirectory() as temporary:
    path = Path(temporary)
    best = path / "best.pt"
    train_text([
        "--corpus", "data/tiny_story.txt",
        "--steps", "20", "--eval-every", "5",
        "--grouped-adamw", "--weight-decay", "0.1",
        "--adam-beta1", "0.9", "--adam-beta2", "0.95",
        "--gradient-accumulation-steps", "3",
        "--grad-clip", "1.0",
        "--warmup-steps", "2", "--lr-decay-steps", "15", "--min-lr", "0.002",
        "--output", str(path / "final.pt"),
        "--best-output", str(best),
        "--history-file", str(path / "history.json"),
    ])
    print("generated text from the saved best model:")
    generate_text([
        "--checkpoint", str(best), "--prompt", "a",
        "--new-tokens", "25", "--top-k", "5", "--seed", "7",
    ])
