# Lesson 216 — Sweep the number of balanced start regions

Region-balanced validation can guarantee near-equal **start counts**
but cannot guarantee how many distinct corpus-token positions are
actually visited. Making the regions finer may reveal new blind
spots, but it may also increase repeated context coverage.

The new model-free `examples/113_balanced_region_sweep.py`
compares multiple region counts at one fixed sample budget and
on one contiguous held-out text split:

```bash
python examples/113_balanced_region_sweep.py \
  --corpus data/tiny_story.txt --region-counts 2,4,8 \
  --batch-size 4 --eval-batches 4 --context 8 \
  --position-bins 8 --csv out/regions.csv \
  --json out/regions.json
```

CSV reports the balance of eligible-start counts, global token
coverage, cross-seed repeated token positions and the least
covered positional bin fraction. JSON adds per-seed counts and
individual corpus-token coverage bins.

When you hold the total window budget fixed, each balanced
region must get at least one start. Asking for more regions
than sampled windows (or eligible start positions) fails explicitly.

This is not a hyperparameter search that chooses the statistically
best estimator. The balanced policy changes the distribution of
starts and the results remain descriptive observations on one
corpus, not independent samples.

Run `python -m pytest tests/test_balanced_region_sweep.py -q`.

**Experiment:** Compare two and eight start regions with the same
seed count. Does token coverage necessarily improve?

**Checkpoint:** Why do counts of eligible starts and counts of
covered corpus positions use different coordinate systems?
