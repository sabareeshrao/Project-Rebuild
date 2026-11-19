# Lesson 213 — Inspect start balance within seeds, not only in aggregate

A pooled start-position histogram can look perfectly balanced
even when **every individual validation seed** misses entire regions.

```text
                  Region A B C D
Seed 1 starts           2 2 0 0
Seed 2 starts           0 0 2 2
Pooled starts           2 2 2 2
```

Our new `summarize_start_regions` reports both levels:
the per-seed region counts and empty-region counts, alongside
aggregate counts, pooled imbalance, and the worst imbalance
of an individual seed.

It uses the same eligible-start boundaries for every seed,
checks that plans share one corpus, and performs no additional
sampling or model evaluation.

A region-balanced sampler guarantees roughly equal **starts**
within each seed when feasible, but pooled balance from ordinary
random draws does not provide that same guarantee. Neither
guarantees equal *token coverage*, independent samples or
an unbiased estimate of GPT held-out loss.

Run `python -m pytest tests/test_start_region_summary.py -q`.

**Experiment:** Reverse the order of two complementary seed
plans. Does either the pooled or per-seed summary change?

**Checkpoint:** Why must evaluation reports include the
worst single-seed imbalance and not just aggregate imbalance?
