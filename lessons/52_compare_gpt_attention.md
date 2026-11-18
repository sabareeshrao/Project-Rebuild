# Lesson 52: switch between educational and fused GPT attention

We can now choose the attention implementation in a GPT configuration:

```python
GPTConfig(..., attention_impl="educational") # original separate heads
GPTConfig(..., attention_impl="fused")       # packed QKV + batched attention
```

Both paths maintain the same external interface and tensor shapes,
and both use causal masking. The default remains **educational** to
preserve previous lessons and their architecture-specific counts.

The fused implementation uses a learned `c_attn` bias as well as
an output `c_proj`, like the GPT-2-style nanoGPT path. Consequently,
each fused transformer block has **3*C extra bias parameters** compared
with our previous separate-head implementation:

```text
default tied model:
educational = V*C + T*C + 2*C + L*(12*C*C + 10*C)
fused       = V*C + T*C + 2*C + L*(12*C*C + 13*C)
```

Both architectures use tied token/output embeddings by default.
GPT initialization also scales the fused attention's `c_proj` as a
residual output projection.

Run: `python examples/52_compare_gpt_attention.py`.

**Experiment:** Keep V, T, C and L fixed, then switch attention
implementations. Why does the trainable parameter count differ by
exactly `3*C*L`?

**Checkpoint:** Why is matching the attention mathematics and QKV
weight layout necessary but insufficient for loading an original
nanoGPT checkpoint without further name/config mapping?

Next milestones include GPT-compatible module naming and saved
checkpoints, realistic data preprocessing, and reproducible generation.
