# Lesson 200 — When does a preselected seed sequence hit a coverage goal?

After measuring the marginal token positions added by each seed,
we can ask when the *already-chosen* seed sequence first covers a
specified fraction of the held-out corpus.

`assess_coverage_target` takes a coverage curve, corpus token
count and target fraction in (0, 1]. It rounds the target up to
a whole token position.

```text
Corpus: 20 token positions, target 60% = 12 positions
seed 10 → 8 unique positions (not reached)
seed 11 → 12 unique positions (reached at seed #2)
seed 12 → 12 unique positions (zero new coverage)
```

The result reports the first seed number and seed ID reaching
the target. If the target was not reached, it reports the remaining
number of positions required rather than inventing additional
seeds or pretending the run succeeded.

**Important:** This is a retrospective experiment-design
diagnostic. It is not a valid statistical stopping rule for
model selection, and token coverage is not equivalent to
independent observations, confidence or generalization.

Run `python -m pytest tests/test_coverage_targets.py -q`.

**Experiment:** Try 40%, 60%, 80% and 100% targets on the same curve.

**Checkpoint:** Why can a high coverage fraction still provide
poorly calibrated evidence about a model's performance on new text?
