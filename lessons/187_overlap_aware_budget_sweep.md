# Lesson 187 — Compare sampling policies across equal validation budgets

Our existing paired-checkpoint budget sweep evaluates two GPT
checkpoints on increasingly many sampled windows. It can now compare
`replacement`, `disjoint`, or `both` policies:

```bash
python examples/110_paired_budget_sweep.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt \
  --sampling-policy both --eval-batch-counts 1,2,4 \
  --csv out/policy_sweep.csv --json out/policy_sweep.json
```

For every sample count and sampling policy, the sweep reports
paired mean differences, descriptive variation across seeds, and
retained overlap statistics (duplicate starts, repeated token
positions and unique positions covered within each seed).

Both policies use the same **number** of token targets and evaluation
seeds. They do *not* necessarily use the same examples: the fixed
disjoint tile grid changes the sampling distribution.

Larger counts within one policy reuse a prefix of selected windows
for the same seed. The rows are therefore dependent, not fresh
independent experiments. Across seeds, identical tiles can still
be drawn; combined overlap totals sum per-seed metrics and do not
globally deduplicate positions.

No policy is declared universally more statistically reliable or
a better generalization estimate. These are reproducible
experiment-design diagnostics.

Run `python -m pytest tests/test_policy_budget_sweep.py -q`.

**Experiment:** Compare two identical checkpoints with both
policies. Why should all paired deltas be zero even when the overlap
statistics differ?

**Checkpoint:** Why isn't a lower overlap count enough by itself
to establish that an evaluation estimator is unbiased?
