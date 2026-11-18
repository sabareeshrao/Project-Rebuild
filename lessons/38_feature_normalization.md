# Lesson 38: LayerNorm's underlying calculation

A transformer passes vectors of hidden features between layers.
**Layer normalization** stabilizes their scale. It computes a mean
and variance over the **feature dimension C**, separately for every
token and every batch row:

```text
mean = mean(x, along C)
variance = mean((x - mean)^2, along C)
normalized = (x - mean) / sqrt(variance + eps)
```

For input `[B,T,C]`, output stays `[B,T,C]`.
Using a small positive epsilon avoids division by zero on constant
vectors. We use the *population* variance, not an unbiased estimate.

Run: `python examples/38_feature_normalization.py`.

**Experiment:** Add 10 to all features of one token. Why does its
normalized output stay the same?

**Checkpoint:** Why is normalizing each token independently important
for autoregressive causality?
