# Lesson 129 — Design a controlled scaling experiment

One benchmark measurement doesn't show what causes a performance
change. Instead, choose a small set of values for each dimension,
then enumerate every combination *deterministically*.

```text
batch sizes = (1, 2)
contexts    = (4, 8)
layers      = (1, 3)
             ↓
2 × 2 × 2 = 8 configurations
```

Our grid preserves the order supplied for each dimension, rejects
duplicate or invalid sizes, and defaults to no more than 64 points.
This prevents an accidentally huge benchmark sweep.

Each point reports the same mathematical quantities from Lesson 128:
tokens per microbatch, effective tokens per optimizer update,
one explicit attention-score tensor and its across-layer sum.

There is **no model allocation or GPU measurement** in this commit.
In the next commit we'll export the grid, then use it to drive small
training benchmarks.

Run `python -m pytest tests/test_scaling_grid.py -q`.

**Experiment:** Sweep B=(1,2), T=(4,8), L=(1,2,4).
How many configurations are generated?

**Checkpoint:** If batch size and context length both double,
what happens to the number of attention-score elements?
