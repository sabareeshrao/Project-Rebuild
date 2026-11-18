# Lesson 46: where do GPT parameters come from?

A parameter is an independently stored model weight or bias.
The same tensor used at many token positions is counted **once**.
Our count includes learned embeddings, attention and MLP weights,
LayerNorm scales/biases, and the output projection.

For our current untied, readable implementation, with vocabulary
size V, context length T, hidden width C, and depth L:

```text
token embeddings       = V*C
position embeddings    = T*C
final LayerNorm        = 2*C
vocabulary projection  = V*C
one transformer block  = 12*C*C + 10*C
total = 2*V*C + T*C + 2*C + L*(12*C*C + 10*C)
```

The block formula includes Q/K/V weights, the multi-head output
projection and its bias, two LayerNorm layers, and both MLP layers.
This is **specific to our current implementation**: nanoGPT's
original model uses some different parameter conventions, including
tied embedding/output weights.

Run: `python examples/46_count_parameters.py`.

**Experiment:** Double L, then double C. Which change grows the
parameter count more dramatically?

**Checkpoint:** Why does increasing the number of transformer blocks
not change the shape of the final logits?
