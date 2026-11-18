# Lesson 41: build a complete pre-LayerNorm transformer block

We now combine **causal multi-head attention** with a position-wise
**GELU MLP**, wrapping both in a normalization-before-operation step
and a residual connection:

```text
x ----------------------------+
  -> LayerNorm -> Attention --+ -> x1
x1 ---------------------------+
   -> LayerNorm -> MLP -------+ -> output
```

Equivalently:

```python
x = x + attention(layer_norm_1(x))
x = x + mlp(layer_norm_2(x))
```

This is called **pre-norm**, because each sublayer sees normalized
features before processing them. The two skip connections help
gradients and original information travel through a stack of blocks.

The causal mask is still inside each attention head. LayerNorm
normalizes features per token, and the MLP processes tokens separately:
neither operation exposes future information.

Run: `python examples/41_transformer_block.py`.

**Experiment:** Zero the attention and MLP parameters. Why does
the block become an identity function?

**Checkpoint:** Why do we apply normalization twice rather than
using one normalization for both sublayers?
