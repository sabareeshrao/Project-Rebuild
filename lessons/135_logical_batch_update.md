# Lesson 135 — One logical batch, many forward/backward passes

We now expose `train_logical_batch_step` as a simple helper:

```python
train_logical_batch_step(
    model, optimizer, inputs, targets, microbatch_size=2,
)
```

When `inputs` contains B=5 examples, this makes three chunks of
sizes 2, 2 and 1. Every chunk participates in backpropagation, but
the optimizer updates parameters **once**, after all gradients are
accumulated and optionally clipped.

```text
logical batch [5 examples]
        ├─ chunk 2 → backward (weight 2/5)
        ├─ chunk 2 → backward (weight 2/5)
        └─ chunk 1 → backward (weight 1/5)
                     ↓
                  one AdamW step
```

When the microbatch can hold the whole logical batch, the helper
delegates to `train_step`, preserving its normal semantics.

Run `python -m pytest tests/test_logical_batch_step.py -q`.

**Experiment:** Change the microbatch size from 5 to 1 and
compare the resulting parameter tensors.

**Checkpoint:** Why does the number of forward passes increase as
the microbatch size shrinks while the number of optimizer updates
remains one?
