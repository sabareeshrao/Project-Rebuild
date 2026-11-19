# Lesson 220 — Pair two GPT models and reweight their sampled losses

We can now compute both the raw balanced-sample difference and
a region-population-weighted estimate of the difference between
two GPT models over *all eligible starts*.

```python
plan = prepare_balanced_evaluation_index_plan(
    validation_tokens, batch_size=3, block_size=4,
    eval_batches=2, seed=226, region_count=4,
)
result = compare_balanced_window_losses(
    baseline, candidate, plan, region_count=4,
)
print(result.raw_paired_delta, result.corrected_paired_delta)
```

Each model receives **the same start positions**. We compute
next-token loss separately per window and then compare candidate
minus baseline at that exact start. The region-population weights
are calculated from how many eligible start positions each region
contains, rather than how many windows happened to be sampled.

Both scores are descriptive observations. The reweighted estimate
is unbiased for a uniform-start mean under uniform-within-region
sampling and deterministic per-start scores, but samples may
overlap, and uncertainty is not calibrated here.

The model's train/eval mode, parameters and the global random
generator state are preserved.

Run `python -m pytest tests/test_stratified_model_comparison.py -q`.

**Experiment:** Compare two identical models on the same index plan.

**Checkpoint:** Why is candidate-minus-baseline pairing performed
at each start *before* aggregating by region?
