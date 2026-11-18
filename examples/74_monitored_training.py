"""Lesson 77: train with history, restore the best model and generate text."""

from pathlib import Path
from tempfile import TemporaryDirectory

from rebuild_gpt.text_cli import main as generate_main
from rebuild_gpt.train_cli import main as train_main

with TemporaryDirectory() as folder:
    root = Path(folder)
    model = root / "model.pt"
    best = root / "best.pt"
    history = root / "history.json"

    train_main([
        "--corpus", "data/tiny_story.txt", "--steps", "30",
        "--eval-every", "10", "--eval-batches", "3",
        "--output", str(model), "--best-output", str(best),
        "--history-file", str(history), "--restore-best",
    ])
    print("generated text from the selected checkpoint:")
    generate_main([
        "--checkpoint", str(best), "--prompt", "a",
        "--new-tokens", "50", "--temperature", "0.8",
        "--top-k", "5", "--seed", "13",
    ])
