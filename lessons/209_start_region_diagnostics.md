# Lesson 209 — Did every region get sampled validation starts?

A balanced-start sampler promises nearly equal counts of **start
positions** across a fixed partition of eligible starts. We can
test that property without conflating it with actual token coverage.

`analyze_start_regions` accepts any index plan and reports
start-region boundaries, per-region counts, empty regions,
minimum/maximum counts and their imbalance.

For 12 starts divided across five regions, balanced sampling gives
counts of `[3,3,2,2,2]`, so no region is empty and the largest
minus smallest count is one. A standard random sampler can, by
chance, concentrate starts in only one region.

Context windows overlap and cross regional boundaries. Thus
zero **empty start regions** does not imply full coverage of any
region's token positions or unbiased validation loss.

```python
report = analyze_start_regions(plan, region_count=5)
print(report.counts, report.empty_regions, report.imbalance)
```

Run `python -m pytest tests/test_start_region_diagnostics.py -q`.

**Experiment:** Measure start-region balance and token-position
coverage on the same sampled validation index plan.

**Checkpoint:** Why should start-position counts and corpus-token
coverage be reported as two separate metrics?
