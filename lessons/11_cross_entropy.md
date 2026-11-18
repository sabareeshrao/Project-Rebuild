# Lesson 11: measure next-token prediction error

Cross-entropy loss rewards models that assign high probability to the
**correct next token**. If the correct token receives probability p, its
loss is `-log(p)`. We average this over all batch/time positions.

```text
logits [B, T, V] -> log-softmax -> select correct token -> mean negative log-prob
targets [B, T] ----------------------^
```

Instead of calculating `log(softmax(logits))` directly, use a stable
log-sum-exp expression:

```text
log p_i = z_i - log(sum_j exp(z_j))
```

Our implementation uses `torch.logsumexp` to avoid overflow. Unlike
the uniform baseline, the loss operation is differentiable: gradients
can flow back to learnable model parameters later.

Run: `python examples/11_cross_entropy.py`

**Experiment:** Increase the correct token's score. Does the loss
increase or decrease?

**Checkpoint:** Why is a lower cross-entropy better, and why must
the target tensor omit the vocabulary axis?
