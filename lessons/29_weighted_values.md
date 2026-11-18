# Lesson 29: attention weights multiply value vectors

The final step in an attention head is a **weighted sum**.
The weights say *how much information to take* from each key position.

```text
attention weights [B,T,S] @ value vectors [B,S,D]
                       = mixed information [B,T,D]
```

For a query that gives two keys weights `[0.25, 0.75]` and values
`[[2, 0], [0, 4]]`, the result is `[0.5, 3.0]`.

Run: `python examples/29_weighted_values.py`

**Experiment:** Give all weight to the first value, then split the
weight equally. How do the outputs change?

**Checkpoint:** Why is this a weighted average when the weights
are nonnegative and sum to one? Why does matmul also allow gradients
to flow through both the weights and values?
