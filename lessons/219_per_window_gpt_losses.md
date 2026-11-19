# Lesson 219 — Compute per-window GPT losses before reweighting regions

The region-size-weighted estimator expects one score for **each
sampled start index**. A mean loss for a mixed-region minibatch
throws away exactly the alignment required for reweighting.

`collect_window_losses` scores every sampled GPT context and
returns a scalar next-token cross-entropy averaged over that
window's T targets. It preserves the sampled order and the
model's original train/eval mode.

```python
report = collect_window_losses(gpt, index_plan)
assert len(report.starts) == len(report.per_window_losses)
```

The mean of all window losses agrees with the ordinary GPT loss
when all windows have the same context length. We verify this
against `collect_split_losses` for both original and balanced
sampling.

Scoring happens in eval mode with gradients disabled. This
collector does not draw new validation windows, alter model
weights or update optimizer state. It does **not** turn
overlapping token windows into independent samples.

Run `python -m pytest tests/test_window_loss_evaluation.py -q`.

**Experiment:** Compare the list of individual window losses to
the original two per-batch means.

**Checkpoint:** Why should stratified reweighting operate on
per-window losses instead of on a mean pooled across regions?
