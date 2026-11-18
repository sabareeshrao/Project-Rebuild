"""Train a tiny character-level GPT directly from a UTF-8 corpus."""

import math
from pathlib import Path

import torch

from rebuild_gpt.best_model import BestValidationModel
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.data import load_corpus, sample_batch, split_train_val
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.evaluation import estimate_train_val_loss
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer
from rebuild_gpt.training import train_step


def train_char_gpt(
    corpus_path: str | Path,
    *,
    steps: int = 60,
    block_size: int = 8,
    n_embd: int = 16,
    n_head: int = 4,
    n_layer: int = 2,
    batch_size: int = 8,
    learning_rate: float = 0.02,
    seed: int = 7,
    eval_interval: int | None = None,
    eval_batches: int = 4,
    history: list[dict[str, float | int]] | None = None,
    restore_best: bool = False,
    best_checkpoint_path: str | Path | None = None,
) -> tuple[GPTLanguageModel, CharTokenizer, dict[str, float]]:
    """Train on contiguous corpus tokens and report fixed-batch validation loss.

    This is a small CPU-oriented learning exercise, not a production trainer.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    if eval_interval is not None and (type(eval_interval) is not int or eval_interval < 1):
        raise ValueError("eval_interval must be a positive integer or None")
    if type(eval_batches) is not int or eval_batches < 1:
        raise ValueError("eval_batches must be a positive integer")
    if history is not None and eval_interval is None:
        raise ValueError("history requires eval_interval")
    if (restore_best or best_checkpoint_path is not None) and eval_interval is None:
        raise ValueError("Best-model tracking requires eval_interval")
    if type(restore_best) is not bool:
        raise ValueError("restore_best must be boolean")
    if history is not None:
        history.clear()

    tokens, tokenizer = load_corpus(corpus_path)
    train, validation = split_train_val(tokens, val_fraction=0.2)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd, n_head=n_head, n_layer=n_layer,
        attention_impl="fused",
    )
    if len(train) <= block_size or len(validation) <= block_size:
        raise ValueError("Corpus is too short for block_size in train/validation")

    torch.manual_seed(seed)
    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    training_rng = torch.Generator().manual_seed(seed)
    val_x, val_y = sample_batch(
        validation, batch_size=batch_size, block_size=block_size,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    model.eval()
    with torch.no_grad():
        before = model.loss(val_x, val_y).item()
    tracker = BestValidationModel() if (restore_best or best_checkpoint_path is not None) else None

    def record_evaluation(step: int) -> None:
        scores = estimate_train_val_loss(
            model, train, validation,
            batch_size=batch_size, block_size=block_size,
            eval_batches=eval_batches, seed=seed + 2,
        )
        if history is not None:
            history.append({"step": step, **scores})
        if tracker is not None and tracker.consider(model, step, scores["validation_loss"]):
            if best_checkpoint_path is not None:
                save_text_checkpoint(model, tokenizer, best_checkpoint_path)

    if eval_interval is not None:
        record_evaluation(0)

    last_loss = float("nan")
    for step in range(1, steps + 1):
        inputs, targets = sample_batch(
            train, batch_size=batch_size, block_size=block_size,
            generator=training_rng,
        )
        last_loss = train_step(model, optimizer, inputs, targets)
        if eval_interval is not None and (step % eval_interval == 0 or step == steps):
            record_evaluation(step)

    if restore_best:
        assert tracker is not None
        tracker.restore(model)
    model.eval()
    with torch.no_grad():
        after = model.loss(val_x, val_y).item()
    metrics = {
        "train_loss_last": last_loss,
        "validation_before": before,
        "validation_after": after,
    }
    if tracker is not None:
        metrics["best_validation_loss"] = tracker.best_loss
        metrics["best_step"] = float(tracker.best_step)
    return model, tokenizer, metrics
