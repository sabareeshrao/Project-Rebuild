# Lesson 99 — Autocast forward passes and scaled gradients

We now support an optional precision mode in both training helpers.
Without arguments, they still execute the original FP32 path.

For BF16, only forward/loss operations run inside autocast, while
model weights and AdamW optimizer states stay at FP32.

CUDA FP16 uses a `GradScaler` to avoid losing tiny gradients:

```text
with autocast: loss = model.loss(x,y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(...)       # gradients are now in normal scale
scaler.step(optimizer)     # skips update if NaN/Inf is detected
scaler.update()            # adapts scale for next step
```

When accumulating microbatches, each loss is divided by the number of
microbatches, all gradients accumulate, and we unscale/clip/update **once**.

Run `python -m pytest tests/test_amp_steps.py -q`.

**Experiment:** Compare BF16 vs FP32 loss and the `dtype` of the
model's weights after one step.

**Checkpoint:** Why does clipping gradients *before* unscaling them
apply the wrong clipping threshold?

The CUDA FP16 path requires supported hardware; CPU-only systems can
verify the BF16 path and simulated scaling order.
