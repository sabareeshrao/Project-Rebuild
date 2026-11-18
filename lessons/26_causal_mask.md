# Lesson 26: the model must not see the future

For autoregressive next-token prediction, the representation at position
`i` must use only positions `j <= i`. Without a **causal mask**,
training could leak the correct answer from later tokens.

```text
allowed key positions for 4 queries:
         0   1   2   3
query 0  ✓   ×   ×   ×
query 1  ✓   ✓   ×   ×
query 2  ✓   ✓   ✓   ×
query 3  ✓   ✓   ✓   ✓
```

Before softmax, assign `-inf` to disallowed future-key scores.
Their exponentials become zero, so they get **exactly zero weight**.
The diagonal is permitted (a token can attend to itself).

This early implementation expects a square self-attention score matrix
`[B,T,T]`; cross-attention and cache-aware masks come later.

Run `python examples/26_causal_mask.py`.

**Experiment:** Remove the mask and inspect which probability entries
can become nonzero.

**Checkpoint:** Why is hiding future positions essential even though
the training targets are already shifted by one token?
