# Lesson 274: GPT batches are parallel *independent* sequences

Mini-batching accelerates inference and training. It must not change
the meaning of a token sequence. A GPT prediction for sequence A
should be the same whether A runs alone or beside B and C.

```text
[A, B, C] -> GPT -> [logits_A, logits_B, logits_C]
 A        -> GPT ->  logits_A (same as above)
 B        -> GPT ->  logits_B (same as above)
 C        -> GPT ->  logits_C (same as above)
```

This invariant catches accidental cross-example attention and mixing
statistics across independent batch rows. We evaluate with dropout
disabled to avoid randomness confusing the comparison.

Run `python examples/167_batch_independence.py`.

The regression test includes a deliberately broken model that
injects a sum across all examples into every prediction.

**Checkpoint:** Why is LayerNorm applied within each sequence's
features, rather than normalizing all examples using a shared batch
mean and variance?
