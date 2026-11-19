# Lesson 182 — How many sampled validation batches are enough?

A tiny sampled validation set can produce noisy differences when
comparing two GPT checkpoints. Increasing the number of evaluation
batches may stabilize the estimated paired delta, but it also
increases evaluation time.

The new sweep varies only the number of sampled windows evaluated
per seed while holding both checkpoints, text corpus, model
configuration, seeds and batch size fixed:

```bash
python examples/110_paired_budget_sweep.py \
  --baseline out/before.pt --candidate out/after.pt \
  --corpus data/tiny_story.txt \
  --eval-batch-counts 1,2,4,8 --seed-count 3 \
  --csv out/paired_sweep.csv --json out/paired_sweep.json
```

The report records the mean signed delta across seeds, descriptive
spread of those seed means, improvement/worsening/tie counts, and
the number of sampled target positions per model.

**Sampling dependence matters:** For a given RNG seed, K=2 uses the
same first batch as K=1, then one additional batch. These sweep
rows are therefore **not independent**. Different start positions
can produce overlapping windows, too.

Neither the standard deviations nor the directions constitute a
significance test or a calibrated confidence interval. This is a
practical experiment-design diagnostic.

Run `python -m pytest tests/test_paired_budget_sweep.py -q`.

**Experiment:** Compare the same checkpoint with itself. Every
budget should produce exactly zero paired deltas.

**Checkpoint:** Why can't we treat the K=1 and K=8 sweep results as
eight independent experiments?
