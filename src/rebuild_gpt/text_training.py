"""Train a tiny character-level GPT directly from a UTF-8 corpus."""

import math
from pathlib import Path

import torch

from rebuild_gpt.best_model import BestValidationModel
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.corpus_evaluation import full_corpus_loss, full_train_val_loss
from rebuild_gpt.data import load_corpus, sample_batch, split_train_val
from rebuild_gpt.devices import resolve_device
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizers import configure_adamw
from rebuild_gpt.schedule import cosine_warmup_lr
from rebuild_gpt.evaluation import estimate_train_val_loss
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.text_resume import load_text_training_state, save_text_training_state
from rebuild_gpt.tokenizer import CharTokenizer
from rebuild_gpt.training import train_accumulated_step, train_step


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
    max_grad_norm: float | None = None,
    warmup_steps: int = 0,
    lr_decay_steps: int | None = None,
    min_lr: float = 0.0,
    validation_mode: str = "sampled",
    grouped_adamw: bool = False,
    weight_decay: float = 0.1,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    gradient_accumulation_steps: int = 1,
    resume_from: str | Path | None = None,
    training_state_path: str | Path | None = None,
    device: str = "cpu",
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
    if max_grad_norm is not None and (
        isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, (float, int))
        or not math.isfinite(max_grad_norm) or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be finite and positive or None")
    if lr_decay_steps is None:
        if warmup_steps != 0 or min_lr != 0.0:
            raise ValueError("lr_decay_steps required for warmup_steps or min_lr")
    else:
        cosine_warmup_lr(
            0, max_lr=learning_rate, min_lr=min_lr,
            warmup_steps=warmup_steps, decay_steps=lr_decay_steps,
        )
    if validation_mode not in ("sampled", "full"):
        raise ValueError("validation_mode must be sampled or full")
    if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if type(grouped_adamw) is not bool:
        raise ValueError("grouped_adamw must be boolean")
    if not grouped_adamw and (weight_decay != 0.1 or adam_betas != (0.9, 0.95)):
        raise ValueError("weight_decay and adam_betas requires grouped_adamw")
    compute_device = resolve_device(device)
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

    # The total optimizer-step target is intentionally omitted from these
    # controls, so training can continue from step N to a larger step M.
    controls = {
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "seed": seed,
        "eval_interval": eval_interval,
        "eval_batches": eval_batches,
        "max_grad_norm": max_grad_norm,
        "warmup_steps": warmup_steps,
        "lr_decay_steps": lr_decay_steps,
        "min_lr": min_lr,
        "validation_mode": validation_mode,
        "grouped_adamw": grouped_adamw,
        "weight_decay": weight_decay if grouped_adamw else None,
        "adam_betas": list(adam_betas) if grouped_adamw else None,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "track_best": bool(restore_best or best_checkpoint_path is not None),
    }

    # Keep pre-existing CPU checkpoints compatible: for CPU there is no
    # additional control. CUDA checkpoints explicitly record the backend.
    if compute_device.type != "cpu":
        controls["device"] = str(compute_device)

    torch.manual_seed(seed)
    model = GPTLanguageModel(config).to(compute_device)
    optimizer = (
        configure_adamw(
            model, learning_rate=learning_rate, weight_decay=weight_decay,
            betas=adam_betas, device_type=compute_device.type,
        )
        if grouped_adamw else torch.optim.AdamW(model.parameters(), lr=learning_rate)
    )
    training_rng = torch.Generator().manual_seed(seed)
    val_x, val_y = sample_batch(
        validation, batch_size=batch_size, block_size=block_size,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    val_x, val_y = val_x.to(compute_device), val_y.to(compute_device)

    tracker = BestValidationModel() if (restore_best or best_checkpoint_path is not None) else None
    saved = None
    start_step = 0
    if resume_from is not None:
        saved = load_text_training_state(
            resume_from, model=model, tokenizer=tokenizer,
            corpus_path=corpus_path, controls=controls,
        )
        start_step = saved["step"]
        if steps <= start_step:
            raise ValueError("steps must exceed the resumed checkpoint step")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        training_rng.set_state(saved["batch_generator_state"])
        torch.set_rng_state(saved["cpu_rng_state"])
        if tracker is not None:
            if saved["best_state"] is None:
                raise ValueError("Checkpoint is missing best-model tracking state")
            tracker.load_state_dict(saved["best_state"])
        if history is not None:
            history.extend(saved["history"])

    if saved is None:
        model.eval()
        with torch.no_grad():
            before = (
                full_corpus_loss(model, validation, block_size=block_size)
                if validation_mode == "full" else model.loss(val_x, val_y).item()
            )
    else:
        before = saved["validation_before"]

    def record_evaluation(step: int) -> None:
        scores = (
            full_train_val_loss(model, train, validation, block_size=block_size)
            if validation_mode == "full"
            else estimate_train_val_loss(
                model, train, validation,
                batch_size=batch_size, block_size=block_size,
                eval_batches=eval_batches, seed=seed + 2,
            )
        )
        if history is not None:
            history.append({"step": step, **scores})
        if tracker is not None and tracker.consider(model, step, scores["validation_loss"]):
            if best_checkpoint_path is not None:
                save_text_checkpoint(model, tokenizer, best_checkpoint_path)

    if eval_interval is not None and saved is None:
        record_evaluation(0)

    last_loss = saved["last_loss"] if saved is not None else float("nan")
    last_learning_rate = (
        saved["last_learning_rate"] if saved is not None else float(learning_rate)
    )
    for step in range(start_step + 1, steps + 1):
        microbatches = []
        for _ in range(gradient_accumulation_steps):
            inputs, targets = sample_batch(
                train, batch_size=batch_size, block_size=block_size,
                generator=training_rng,
            )
            microbatches.append((inputs.to(compute_device), targets.to(compute_device)))
        if lr_decay_steps is not None:
            last_learning_rate = cosine_warmup_lr(
                step - 1, max_lr=learning_rate, min_lr=min_lr,
                warmup_steps=warmup_steps, decay_steps=lr_decay_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = last_learning_rate
        if gradient_accumulation_steps == 1:
            inputs, targets = microbatches[0]
            last_loss = train_step(
                model, optimizer, inputs, targets, max_grad_norm=max_grad_norm,
            )
        else:
            last_loss = train_accumulated_step(
                model, optimizer, microbatches, max_grad_norm=max_grad_norm,
            )
        if eval_interval is not None and (step % eval_interval == 0 or step == steps):
            record_evaluation(step)

    # Save the most recent optimizer weights BEFORE optionally switching to
    # best-validation weights for inference. Otherwise resume would diverge.
    if training_state_path is not None:
        save_text_training_state(
            training_state_path,
            model=model, tokenizer=tokenizer, optimizer=optimizer,
            corpus_path=corpus_path, controls=controls, step=steps,
            batch_generator=training_rng,
            history=history if history is not None else [],
            validation_before=before, last_loss=last_loss,
            last_learning_rate=last_learning_rate,
            best_state=tracker.state_dict() if tracker is not None else None,
        )

    if restore_best:
        assert tracker is not None
        tracker.restore(model)
    model.eval()
    with torch.no_grad():
        after = (
            full_corpus_loss(model, validation, block_size=block_size)
            if validation_mode == "full" else model.loss(val_x, val_y).item()
        )
    metrics = {
        "train_loss_last": last_loss,
        "validation_before": before,
        "validation_after": after,
    }
    if tracker is not None:
        metrics["best_validation_loss"] = tracker.best_loss
        metrics["best_step"] = float(tracker.best_step)
    if lr_decay_steps is not None:
        metrics["learning_rate_last"] = last_learning_rate
    if grouped_adamw:
        metrics["optimizer_group_count"] = float(len(optimizer.param_groups))
    if gradient_accumulation_steps > 1:
        metrics["effective_tokens_per_step"] = float(
            gradient_accumulation_steps * batch_size * block_size
        )
    return model, tokenizer, metrics
