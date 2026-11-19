# Lesson 179 — Count paired improvements without claiming significance

A model may improve its mean sampled loss while worsening some
individual batches. To understand that mixture, summarize the sign
of the **candidate-minus-baseline** differences:

- Negative: lower sampled loss for the candidate (improved).
- Positive: higher sampled loss (worsened).
- Zero: exactly tied, or within an explicitly chosen tolerance.

`summarize_paired_deltas` computes counts at two levels:
all individual batches and the *per-seed average* of those batches.
These answers can disagree because one big worsening can outweigh
several small improvements.

A `tie_tolerance` can mark tiny differences as ties; the default
is exactly zero, avoiding an arbitrary hidden threshold.

```python
summarize_paired_deltas([[-1, 0, 0.5], [2, -0.5, -0.5]])
```

This is **descriptive accounting**. Overlapping windows and repeated
seeds from one corpus are not independent Bernoulli trials.
Do not attach binomial p-values or interpret improvement fractions
as calibrated success probabilities.

Run `python -m pytest tests/test_paired_diagnostics.py -q`.

**Experiment:** Construct two seeds with identical average delta,
but different numbers of improved/worsened batches.

**Checkpoint:** Why is counting negative differences useful even
though it cannot establish statistical significance?
