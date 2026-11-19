# Lesson 197 — How much new coverage does an extra seed provide?

Several validation seeds are not several independent held-out
datasets. Each seed selects a set of windows on the **same** corpus,
so their token coverage can overlap even if each plan uses
nonoverlapping windows internally.

The new model-free CPU experiment changes the number of seed plans
while keeping the original shared corpus, batch shape, context,
and number of evaluation batches per seed fixed.

```bash
python examples/111_seed_coverage_sweep.py \
  --corpus-tokens 2048 --batch-size 4 --context 16 \
  --seed-counts 1,2,4,8 --sampling-policy all \
  --csv out/seed_coverage.csv --json out/seed_coverage.json
```

We record global distinct token coverage, the fraction of corpus
touched, repeated positions inside each seed and additional
repeated positions between seeds. Growing the number of seeds
retains the original sampled plans as a prefix, so **union coverage
cannot decrease**. However, its marginal gain can shrink as the
same positions are touched repeatedly.

All samples still come from one synthetic token stream. No model
losses, optimizer steps, or calibrated confidence intervals are
involved, and low overlap alone does not establish independence.

Run `python -m pytest tests/test_seed_coverage_sweep.py -q`.

**Experiment:** Plot global coverage versus seed count. Compare
replacement, fixed-origin disjoint and phase-shifted disjoint modes.

**Checkpoint:** Why should a new validation seed's useful
contribution be measured by *additional unique positions*, not
just the total number of token windows it sampled?
