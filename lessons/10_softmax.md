# Lesson 10: logits → probabilities with softmax

Prediction scores need not be positive or sum to 1. To turn logits
`z_i` into a categorical distribution, use

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

Exponentials of large logits can overflow. Subtracting the **maximum
logit** before exponentiation changes no probabilities:

```text
softmax(z) = softmax(z - max(z))
```

For model predictions `[B, T, V]`, softmax operates independently across
the last axis, the `V` candidate next tokens.

Run: `python examples/10_softmax.py`

**Experiment:** Add 100 to every logit. Why are probabilities unchanged?

**Checkpoint:** Explain why the maximum-subtraction trick works
mathematically and what happens when all logits are equal.
