# Lesson 210 — Pair two GPTs on region-balanced validation starts

The paired-comparison API now accepts a fourth opt-in
`sampling_policy="balanced"`, plus a configurable
`sampling_regions` count.

```python
result = compare_paired_batches(
    baseline, candidate, heldout_tokens,
    seeds=(1, 2, 3), batch_size=4, block_size=8,
    eval_batches=4, sampling_policy="balanced",
    sampling_regions=4,
)
```

For each seed, one balanced index plan is shared by both models,
so the paired candidate-minus-baseline losses are evaluated on
exactly the same token windows. The result records start counts,
empty region counts and imbalance, separately from the existing
corpus coverage and overlap metrics.

A balanced-start plan does **not** have a nonoverlapping lattice;
the lattice metadata is explicitly null. Starts can repeat and
context windows can overlap, making equal start counts different
from equal token coverage.

The saved-checkpoint comparator accepts the same opt-in policy.
Older callers still default to uniform random starts with
replacement and return identical paired losses.

Run `python -m pytest tests/test_balanced_paired_evaluation.py -q`.

**Experiment:** Compare an identical saved checkpoint with itself.
Why must every paired delta be zero even when windows overlap?

**Checkpoint:** Why is pairing samples between models more important
for interpreting *relative* loss than reducing start imbalance alone?
