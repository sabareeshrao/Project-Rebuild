# Lesson 74: track loss during training instead of once at the end

The original tiny training example only measured validation loss
before and after training. That hides what happened in between.

With an optional `eval_interval`, the trainer now evaluates at
**step 0**, every interval, and the **final step**, recording both
train and validation estimates:

```text
step 0  -> loss on fixed evaluation batches
step 10 -> loss on the same evaluation batches
step 20 -> ...
step 30 -> ...
```

Pass an empty `history` list to collect simple records that can be
plotted or serialized later. The original return shape and the default
training behavior remain unchanged for earlier lessons.

Sampling evaluation batches uses a separate RNG stream, and
evaluation disables dropout and gradient computation, so logging
does **not** alter the sequence of training updates.

Run: `python examples/71_training_history.py`.

**Experiment:** Compare curves for different `eval_interval` and
`eval_batches` values. What is the cost of evaluating frequently?

**Checkpoint:** What does it mean when training loss keeps decreasing
while validation loss starts rising?
