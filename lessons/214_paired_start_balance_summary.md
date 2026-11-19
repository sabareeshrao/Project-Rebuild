# Lesson 214 — Carry the whole seed-level balance audit into GPT comparisons

We can now compare two GPT checkpoints on exactly the same
region-balanced starts and inspect balance in **two ways**:
counts inside each seed and pooled counts across all seeds.

`compare_paired_batches` returns an
`aggregate_start_regions` summary for the balanced policy.
It includes regional counts, total sampled windows, empty regions
per seed, worst per-seed imbalance and pooled imbalance.

```python
result = compare_paired_batches(
    baseline, candidate, heldout_tokens,
    seeds=(1, 2, 3), batch_size=3, block_size=5,
    eval_batches=4, sampling_policy="balanced",
    sampling_regions=5,
)
print(result.aggregate_start_regions.aggregate_counts)
```

The ordinary replacement, fixed-disjoint and shifted policies
continue to return `None` for this extra field and **do not
acquire any minimum-region requirement**. This matters for small
held-out corpora.

The audit is computed from the already-sampled plans; it runs
no additional GPT forward passes and consumes no new random
draws. Every paired delta remains tied to the same samples.

Run `python -m pytest tests/test_paired_start_region_summary.py -q`.

**Experiment:** Compare a pooled histogram against each seed's
histogram with strongly unequal samples.

**Checkpoint:** Why is equal start count per region still
compatible with repeated input/target token positions?
