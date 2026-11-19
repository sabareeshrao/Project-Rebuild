# Lesson 245 — Examine AdamW's internal state

AdamW is *stateful*. For each parameter it maintains a moving average
of past gradients (first moment), a moving average of their squares
(second moment), and a step counter.

```text
gradient g_t
  exp_avg    <- beta1 * exp_avg    + (1-beta1) * g_t
  exp_avg_sq <- beta2 * exp_avg_sq + (1-beta2) * g_t²
  step       <- step + 1
```

Our named snapshots clone these tensors, so an additional optimizer
step cannot mutate a previously recorded observation. We compare by
parameter names, not the order optimizer groups happen to use.

Run `python examples/138_adamw_moment_snapshot.py` after creating
the capstone checkpoint.

**Checkpoint:** Why can two models with equal weights still take
different next optimizer steps if their AdamW moments differ?
