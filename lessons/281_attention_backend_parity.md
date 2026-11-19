# Lesson 281: manual attention and SDPA should agree

Our GPT supports two packed-QKV causal attention paths:

- **fused:** compute scaled QK scores, causal mask, softmax, V mixing.
- **sdpa:** use PyTorch's scaled-dot-product attention primitive.

When given the **same weights, tokens and masked targets**, both
should produce nearly identical logits, loss and gradients.

```text
identical GPT weights + same tokens
      |                         |
 manual QK-softmax-V           PyTorch SDPA
      |                         |
   logits + loss             logits + loss
   backward                  backward
      +--------- compare gradients -------+
```

This exercise constructs two disposable CPU models, loads identical
state dictionaries, disables dropout and compares their tensors
within numerical tolerance. Your original GPT remains unchanged.

Run `python examples/174_attention_backend_parity.py`.

**Checkpoint:** Why are small floating-point tolerances appropriate
when mathematically equivalent attention implementations use
different operator kernels?
