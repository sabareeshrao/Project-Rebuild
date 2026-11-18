# Lesson 23: how much does each query match each key?

A **dot product** multiplies vector components pairwise and adds them.
It measures alignment, but is **not** cosine similarity unless vectors
are normalized. Larger vector magnitudes can give larger scores.

With `Q` of shape `[B,T,D]` and `K` of shape `[B,S,D]`:

```text
Q [B,T,D] @ transpose(K) [B,D,S] -> scores [B,T,S]
```

Each score `scores[b, i, j]` compares query `i` with key `j`.
**No softmax, future-token masking, or value vectors yet.**

Run `python examples/23_pairwise_dot_scores.py`.

**Experiment:** Double a query vector. What happens to the row of scores?

**Checkpoint:** Why do we transpose keys rather than multiply
`Q @ K` directly?
