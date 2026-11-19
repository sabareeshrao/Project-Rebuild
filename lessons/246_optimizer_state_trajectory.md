# Lesson 246 — Match the hidden AdamW state at EVERY step

At each training step we already compare loss, gradients, and updated
model parameters. Now we also compare the internal first/second
moment tensors and per-parameter step counters.

```text
identical batch t
  | -> local backward   -> local AdamW state
  | -> upstream backward -> upstream AdamW state
  +--------------------- compare -------------------+
            weights / exp_avg / exp_avg_sq / step
```

The same optimizer objects are reused across all steps. Different
batches change the gradients, while AdamW carries a history of earlier
gradients. If the moment state drifts, a later step may diverge even
when current model parameters still look close.

An opt-in example compares against a separately supplied, reviewed
original nanoGPT `model.py`:
```bash
python examples/139_optimizer_moment_trajectory.py \
  --upstream-model /trusted/nanoGPT/model.py
```

**Question:** Which extra state does AdamW remember that plain
stochastic gradient descent does not?
