# Lesson 128 — Distinguish batch size, context, depth, and accumulation

Three independent dimensions affect different parts of GPT training:

- **Batch size B**: how many sequences are processed at once.
- **Context length T**: tokens per sequence (and quadratic attention scores).
- **Depth L**: number of transformer blocks.
- **Accumulation A**: consecutive microbatches contributing to one update.

The *effective tokens per optimizer update* are `B × T × A`.
One explicit attention-score matrix is `B × H × T × T` elements,
and counting one such matrix in each of L blocks gives a mathematical
sum proportional to `B × H × T² × L`.

```text
microbatch:      B × T tokens
optimizer step:  B × T × A tokens
one score:       B × H × T × T elements
all L scores:    L × B × H × T × T elements
```

**Important:** Sum of score-matrix sizes is *not peak GPU memory*.
Autograd buffers, residual paths and optimizer state change the
actual peak; optimized SDPA may not materialize score matrices.
Accumulation runs microbatches sequentially, so it does not necessarily
multiply the simultaneous activation memory by A.

Run: `python -m pytest tests/test_workload_scaling.py -q`.

**Experiment:** Hold B=2, H=4, L=2 constant and double T. Then hold
T constant and double accumulation. Which quantities change?

**Checkpoint:** Why can gradient accumulation increase effective
tokens per update without needing the same activation memory as one
giant batch?
