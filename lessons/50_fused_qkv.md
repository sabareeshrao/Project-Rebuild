# Lesson 50: one projection for all Q, K and V

In the earlier educational attention module, every head contained
three independent `Linear(C,D)` operations. That was easy to
understand but requires many small matrix multiplications.

A GPT-2-style layer combines them into a single
`Linear(C, 3*C)` projection, with learnable weights and biases.
The resulting last dimension contains all query, key and value
features in three contiguous groups:

```text
hidden [B,T,C] -> c_attn [B,T,3C]
                  -> split Q/K/V [B,T,C] each
                  -> reshape heads [B,H,T,D] each; H*D=C
```

This is an efficient **fused projection**, not a different attention
algorithm. These Q/K/V values are not yet multiplied in this lesson.

Run: `python examples/50_fused_qkv.py`.

**Experiment:** Double `n_head` while holding C fixed and divisible.
How does D change? Does `c_attn.weight` change shape?

**Checkpoint:** Why is the packed tensor split into Q, K and V
before computing attention rather than used as one 3C-wide vector?

The fused format moves us closer to nanoGPT's `c_attn` layout, but
we have not claimed checkpoint compatibility yet.
