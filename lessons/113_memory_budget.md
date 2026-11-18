# Lesson 113 — Estimate the persistent tensor memory of GPT

Before profiling GPU memory, understand where training storage comes from:

- **Weights**: each unique parameter tensor (tied embeddings count once).
- **Buffers**: non-parameter tensors registered on the model.
- **Gradients**: an estimate of one gradient tensor per trainable weight.
- **AdamW**: an estimate of two additional moment tensors per trainable weight.

For a trainable FP32 parameter with N elements, these typically total
`4N + 4N + 8N = 16N` bytes *before* activations and allocator overhead.
Frozen parameters still consume storage but need no gradient/moments.

Our estimator counts actual dtypes, not an assumed fixed four-byte width.
It reports **expected storage**, NOT measured peak RAM/VRAM. Real
allocations also include intermediate activations, temporary workspaces,
optimizer scalars, PyTorch allocator caches and possible framework overhead.

Run `python -m pytest tests/test_memory.py -q`.

**Experiment:** Tie an output layer's weights to an input embedding.
How many bytes disappear from the parameter estimate?

**Checkpoint:** Why does batch size affect activation memory but not the
number of stored model parameters?
