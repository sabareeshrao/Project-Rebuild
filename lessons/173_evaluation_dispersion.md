# Lesson 173 — A single validation loss hides sample variability

Sampled validation evaluates GPT on a fixed number of randomly
selected token windows. A mean loss is useful, but different
windows may have noticeably different difficulty.

`summarize_losses` reports the number of observations, arithmetic
mean, minimum, maximum, sample standard deviation (denominator
`n-1`), and standard error `s / sqrt(n)`.

For just one observation, sample standard deviation and standard
error are **undefined**; the helper returns `None` rather than zero.

Important limitation: token windows from the same corpus can overlap,
so per-batch losses are not guaranteed independent observations.
These statistics describe the sample; the standard error is **not**
automatically a valid confidence interval for generalization.

```bash
python -m pytest tests/test_evaluation_stats.py -q
```

**Experiment:** Compare `[1,2,3]` with `[2,2,2]`. Both have the
same mean. How different is their measured dispersion?

**Checkpoint:** Why shouldn't a single validation batch be assigned
a standard error of zero?
