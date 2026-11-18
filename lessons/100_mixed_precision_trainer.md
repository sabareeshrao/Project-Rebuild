# Lesson 100 — Train a character-level GPT using mixed precision

The whole training pipeline now accepts a `precision` setting:

```python
train_char_gpt("data/tiny_story.txt", steps=2, precision="bf16", device="cpu")
```

This switches **forward and validation calculations** into autocast,
while model parameters and optimizer state remain FP32. For supported
CUDA hardware, FP16 uses dynamic gradient scaling, including accumulated
microbatches and clipping after unscale.

```text
CPU token stream -> x,y -> CPU or CUDA model
                             |
                 autocast(forward + loss)
                             |
                  scale/backward/clip/step
                             |
                  eval with same autocast
```

The default `precision="fp32"` preserves the previous code path and
existing FP32 training-state metadata. BF16 run metadata records the
precision so an incompatible checkpoint cannot be loaded by mistake.

At this stage FP16 **training-state saves/resume are rejected** because
they also need the scaler's state; that is the next lesson.

Run `python -m pytest tests/test_mixed_precision_training.py -q`.

**Checkpoint:** Why is it misleading to autocast training but validate
the resulting model using a different precision without documenting it?
