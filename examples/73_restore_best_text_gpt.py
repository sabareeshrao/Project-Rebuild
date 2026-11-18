"""Lesson 76: save and restore the checkpoint with lowest held-out loss."""

from pathlib import Path
from tempfile import TemporaryDirectory

from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.text_training import train_char_gpt

with TemporaryDirectory() as folder:
    best_path = Path(folder) / "best.pt"
    history: list[dict[str, float | int]] = []
    model, tokenizer, metrics = train_char_gpt(
        "data/tiny_story.txt", steps=30, eval_interval=10,
        eval_batches=3, history=history,
        best_checkpoint_path=best_path, restore_best=True,
    )
    loaded, saved_tok = load_text_checkpoint(best_path)
    print("best step:", int(metrics["best_step"]))
    print("best held-out estimate:", round(metrics["best_validation_loss"], 4))
    print("bundle vocabulary matches:", saved_tok.vocab == tokenizer.vocab)
    print("loaded config matches:", loaded.config == model.config)
