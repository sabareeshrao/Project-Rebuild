# Lesson 218 — Correct a balanced sample to estimate uniform-start mean

A region-balanced evaluator samples roughly equal numbers of valid
window starts per region. The original GPT validation objective,
however, samples *all eligible starts uniformly*.

If regions have different populations of eligible starts or
different sampled counts, the raw average of sampled scores may
not give the same weight to every eligible start.

For region r, let M_r be its eligible-start population and
mean_r the mean of its **per-window** observed scores. Then:

```text
uniform-start estimate = sum_r (M_r / sum_j M_j) * mean_r
```

`estimate_start_uniform_score` computes this number alongside the
raw sample mean and each region's population size. It requires at
least one observed score per region and validates exact alignment
between individual starts and individual scores.

This estimate is unbiased for the uniform-start population mean
*if* samples are uniform within each region and per-window scores
are deterministic for those starts. Neither window dependence nor
small numbers of samples permit calibrated confidence intervals.

We also fix a checkpoint-sweep regression: a custom balanced
region count must not be sent to replacement, disjoint or shifted
samplers, which reject that irrelevant setting.

Run `python -m pytest tests/test_start_uniform_estimator.py \
tests/test_extended_budget_region_forwarding.py -q`.

**Checkpoint:** Why is an average of batch losses insufficient to
reweight individual starts if a batch contains several regions?
