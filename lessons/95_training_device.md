# Lesson 95 — Move model and training batches onto CUDA

The GPU can execute tensor operations only when their operands are on
the correct device. The corpus and its seeded sampling generator stay on
CPU, but the GPT model and each sampled `(inputs, targets)` batch move
to the selected device.

```text
corpus on CPU -> seeded window sampling -> x,y on CPU
                                                   |
                                          .to(compute_device)
                                                   |
                                GPT on compute_device -> loss -> AdamW
```

The function now accepts `train_char_gpt(..., device="cpu"|"cuda"|"auto")`.
The default stays CPU. On CUDA, grouped AdamW can use the available fused
optimizer implementation. Full-corpus evaluation already places inputs
and targets on the model device.

For compatibility with earlier CPU-only training states, CPU runs keep
their existing control metadata; CUDA runs record their device choice.
This prevents silently resuming a GPU run as a CPU run.

Run: `python -m pytest tests/test_text_training_devices.py -q`

The CUDA test is skipped when no compatible NVIDIA GPU is present.

**Experiment:** Inspect `next(model.parameters()).device` after
selecting CPU. Explain why merely moving the model without moving
batches fails.

**Checkpoint:** Why is CPU-side sampling useful even when weights are on CUDA?
