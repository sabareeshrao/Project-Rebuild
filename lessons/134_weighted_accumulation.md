# Lesson 134 — Why unequal microbatches must be weighted by tokens

Suppose five examples are split into microbatches of sizes 2, 2, 1.
A typical cross-entropy function returns the **mean loss per target**.

An unweighted average of the three microbatch means would give the
one-example microbatch 1/3 of the update: far too much influence.
Instead, weight each microbatch loss by its target count:

```text
L = (2/5)·L_1 + (2/5)·L_2 + (1/5)·L_3
```

This is the same objective as computing the mean loss on the full
five-example batch, provided every target has equal weight.

Our `train_accumulated_step` now handles unequal leading batch
sizes while insisting that trailing input and target dimensions
match. It still calls the optimizer **exactly once** after the
microbatch gradients have been accumulated, including clipping
after all backward passes. The original equal-size arithmetic stays
the same to preserve prior behavior.

The weighting counts `targets.numel()`. For a task with ignored
targets or custom loss reduction, we would need a different
valid-target count, so this helper assumes a mean over all elements.

Run `python -m pytest tests/test_uneven_microbatch_gradients.py -q`.

**Experiment:** Split five examples into (2, 2, 1); compare a
single-batch SGD update to a weighted microbatch update.

**Checkpoint:** Why is averaging the microbatch means equally wrong
when their batch sizes differ?
