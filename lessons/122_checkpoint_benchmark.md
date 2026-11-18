# Lesson 122 — Measure checkpointing's compute/memory tradeoff

Activation checkpointing reduces saved forward activations by
recomputing transformer blocks during backward. This can reduce GPU
memory requirements, but it typically **increases computation**.

We now benchmark two independent tiny GPT training runs:

- Identical model architecture, token batches and starting weights.
- Normal block execution vs block activation checkpointing.
- Complete training steps (forward, backward and AdamW).
- Identical warmup and timing procedures for both options.
- CUDA peak allocator *deltas* only when CUDA is available.

```bash
python examples/93_activation_checkpoint_benchmark.py --device cpu
# With a compatible NVIDIA GPU:
python examples/93_activation_checkpoint_benchmark.py \
  --device cuda --precision bf16 --n-layer 4
```

We report each option's median training-step latency and their ratio,
rather than claiming a predetermined speedup. On CPU we deliberately
report that **CUDA memory cannot be measured**.

The CUDA peak includes allocations managed by PyTorch beyond a
baseline established *after* optimizer state has been initialized.
This is not full GPU VRAM and is sensitive to allocator state and
execution environment.

Run `python -m pytest tests/test_checkpoint_benchmark.py -q`.

**Experiment:** Increase `--n-layer` from 1 to 4 and compare the
checkpointing latency penalty and CUDA memory delta.

**Checkpoint:** Why might slower training steps still make
activation checkpointing beneficial when GPU VRAM is limited?
