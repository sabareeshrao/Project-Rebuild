# Lesson 79: warm up, then decay learning rate

A constant learning rate can be too aggressive at the beginning or
near the end of training. nanoGPT uses a three-stage function,
indexed by the **zero-based optimizer iteration**:

1. Linear warmup for `iteration < warmup_steps`, using
   `max_lr * (iteration + 1) / (warmup_steps + 1)`.
2. Cosine decay toward `min_lr` between warmup and decay endpoints.
3. Hold `min_lr` after `decay_steps`.

```text
lr = min_lr + 0.5*(1 + cos(pi*progress))*(max_lr - min_lr)
progress = (iteration - warmup_steps)/(decay_steps - warmup_steps)
```

This follows original nanoGPT's `get_lr` warmup convention,
including its `iteration + 1` numerator.

The function is pure: it doesn't modify an optimizer. This makes
the learning-rate curve easy to test independently.

Run: `python examples/76_cosine_schedule.py`.

**Experiment:** Plot or print the values at iterations 0 through 8.
Where does the curve reach its maximum and minimum?

**Checkpoint:** Why might a large learning rate be helpful early in
training but harmful after the model has mostly converged?
