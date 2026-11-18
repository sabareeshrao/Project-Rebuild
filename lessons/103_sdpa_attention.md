# Lesson 103: optimized causal attention with PyTorch SDPA

The original readable path computes a full causal attention score matrix:

```text
Q @ Kᵀ / √d → causal mask → softmax → dropout → @ V
```

PyTorch's `scaled_dot_product_attention` (SDPA) accepts `Q, K, V`
directly and may select optimized kernels on supported hardware.
We add `use_sdpa=True` to the existing fused attention module.

**The module parameters do not change.** Only how the attention
operation is computed changes. Existing manual-model checkpoints
remain compatible with the SDPA module's `state_dict`.

We pass `is_causal=True` and, critically, pass `dropout_p=0.0`
during evaluation. Unlike `nn.Dropout`, SDPA applies dropout
whenever its explicit `dropout_p` is nonzero.

Run `python -m pytest tests/test_sdpa_attention.py -q`.

**Experiment:** Compare the outputs with the same weights for context
lengths 1, 2, and 7.

**Checkpoint:** Why is the optimized path *not guaranteed* to be faster
on every CPU/GPU, context length, or precision?
