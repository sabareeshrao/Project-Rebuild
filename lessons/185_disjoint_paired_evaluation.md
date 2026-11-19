# Lesson 185 — Pair both models on the same disjoint tile sample

We now support two sampling policies for paired GPT comparisons:

- `replacement` (default): the original uniform random start draw,
  where windows can overlap or repeat.
- `disjoint`: seeded selection of nonoverlapping T+1-token tiles,
  without replacement *within each seed's plan*.

```python
compare_paired_batches(
    baseline, candidate, val_tokens,
    seeds=(1,2,3), batch_size=4, block_size=8,
    eval_batches=2, sampling_policy="disjoint",
)
```

Both models still share the same index plan. The result now records
the policy and per-seed window-overlap diagnostics, so researchers
can verify that a disjoint plan actually has zero repeated token
positions. The saved inference-checkpoint comparison accepts the
same option.

This does **not** mean disjoint sampling provides an unbiased
estimate of the original random-start validation objective. It
changes the distribution by sampling from a fixed tile grid. Seeds
may share tiles, and all windows still come from one held-out corpus.

Tests prove that the existing default comparison is unchanged and
that disjoint plans fail explicitly when too many windows are needed.

Run `python -m pytest tests/test_disjoint_paired_validation.py -q`.

**Experiment:** Compare the same model to itself under both policies.

**Checkpoint:** How can pairing improve fairness between two models
while leaving sampling-distribution bias unresolved?
