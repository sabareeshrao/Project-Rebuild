# Lesson 51: calculate every attention head together

The earlier implementation constructs each head as an independent
Python module. We can calculate the same causal attention in parallel:

```text
hidden [B,T,C] -> packed Q,K,V [B,H,T,D]
                       |
         (Q @ K^T) / sqrt(D) -> scores [B,H,T,T]
                       |
             future mask -> softmax -> attention dropout
                       |
                    weights @ V -> [B,H,T,D]
                       |
               transpose + reshape -> [B,T,C]
                       |
                output c_proj -> residual dropout
```

This also introduces dropout on the **attention probabilities**, in
addition to output dropout. The dropout is active only during training.

Our tests compare results against
`torch.nn.functional.scaled_dot_product_attention` with dropout off.
They also copy the exact weights from the earlier separate-head
implementation and verify matching outputs within numerical tolerance.

Run: `python examples/51_fused_causal_attention.py`.

**Experiment:** Increase the number of heads while keeping the
embedding width fixed. Why do Q/K/V maintain shape `[B,H,T,D]`
with H*D=C?

**Checkpoint:** Why must the attention softmax normalize along the
**last** axis, the key positions?

This is a learning-oriented fused attention path: it is still not
the FlashAttention acceleration used in optimized nanoGPT runs.
