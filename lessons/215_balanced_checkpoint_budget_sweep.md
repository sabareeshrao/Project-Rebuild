# Lesson 215 — Compare regional balance across saved-checkpoint budgets

The saved-checkpoint evaluation budget sweep now supports
`--sampling-policy balanced` or the four-policy
`--sampling-policy extended` setting.

```bash
python examples/110_paired_budget_sweep.py \
  --baseline out/before.pt --candidate out/after.pt \
  --corpus data/tiny_story.txt \
  --sampling-policy extended --sampling-regions 2 \
  --batch-size 2 --eval-batch-counts 1,2,4 \
  --csv out/balanced_budget.csv --json out/balanced_budget.json
```

For balanced rows, CSV includes aggregate per-region start counts,
the worst per-seed imbalance and the number of seeds that missed
at least one start region. These metrics sit alongside paired
model-loss changes and global token coverage.

**Budget-prefix caveat:** unlike the original replacement sampler,
the balanced policy allocates requested starts by region before
drawing. Increasing the evaluation budget may alter every region's
draw count, so larger sample sets do not necessarily extend the
smaller set as a prefix. The JSON records this explicitly.

The old `all` option continues to mean three policies, and
`both` continues to mean the earlier two. The existing
replacement default and checkpoint comparison policy do not change.

Run `python -m pytest tests/test_balanced_budget_sweep.py -q`.

**Experiment:** Compare start imbalance at two budgets while
keeping models and the held-out corpus fixed.

**Checkpoint:** Why can't we interpret a lower imbalance in
the balanced row as a guaranteed improvement in held-out loss?
