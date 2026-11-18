# Lesson 25: why attention scores are divided by sqrt(D)

A dot product adds **D** component products. Under simple independent,
unit-variance assumptions, the dot product's variance grows roughly
with D. Large score magnitudes can make softmax saturate, producing
small gradients for most candidates.

Scaled dot-product attention uses:

```text
scores = (Q @ K.transpose(-2,-1)) / sqrt(D)
```

The result is still `[B,T,S]`, but score magnitudes are more controlled.
This improves training behavior; it does **not** normalize Q or K.

Run `python examples/25_scaled_dot_scores.py`.

**Experiment:** Compare the softmax outputs before and after scaling,
keeping the queries and keys unchanged.

**Checkpoint:** Why divide by the square root of head size rather
than by sequence length?
