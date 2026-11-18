# Lesson 101 — Resume FP16 training by restoring the GradScaler

FP16 gradient scaling has its own dynamically changing state: current
scale, growth tracker, and growth/backoff settings. Saving only model
weights and AdamW moments would not reproduce FP16 update decisions.

The resumable checkpoint now includes `grad_scaler_state`. On restart:

1. Validate corpus, tokenizer, GPT config and precision controls.
2. Load model and optimizer weights.
3. Restore the GradScaler when precision is FP16.
4. Restore CPU/batch/CUDA RNG streams.
5. Continue at the saved *optimizer-update attempt* step.

For FP32/BF16, the scaler state remains `None`; previous checkpoints
without this optional field also remain readable.

```text
FP16 state = model + AdamW + GradScaler + CPU/GPU RNG + progress
```

Run: `python -m pytest tests/test_text_resume.py -q`.

**Experiment:** Inspect the scaler state before and after several FP16
iterations on a GPU. How does it respond to a nonfinite gradient?

**Checkpoint:** Why might an FP16 run skip an optimizer update when
the gradients overflow? Can a skipped update change the intended
learning-rate schedule?

We still do not promise bitwise equality across CUDA hardware and
PyTorch versions. Distributed training requires additional state.
