# Lesson 08: predict without learning

Our first "model" ignores the context and assigns score zero to every
possible next token. That is a **baseline**, not a useful GPT.

For inputs of shape `(B, T)` and a vocabulary with `V` tokens, prediction
scores (called **logits**) have shape `(B, T, V)`.

```text
inputs: [B, T] -> baseline -> logits: [B, T, V]
                             last axis = candidate next tokens
```

A logit is an unconstrained real-valued score, **not** a probability.
Equal logits imply equal probabilities after softmax.

Run: `python examples/08_uniform_baseline.py`

**Experiment:** Increase vocabulary size from 4 to 7. Which dimension
changes? Does changing an input token affect any predicted score?

**Checkpoint:** Why is a uniform baseline useful even though it cannot
learn anything?
