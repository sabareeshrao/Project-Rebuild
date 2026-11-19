# Lesson 174 — Expose the losses hidden by a mean

Each sampled evaluator previously returned just an arithmetic mean.
That is the correct quantity for model selection, but it hides the
individual sampled losses used to calculate it.

`collect_split_losses` returns a tuple of per-batch losses for
each evaluation backend: **regenerated**, **cached token windows**
and **cached start indices**.

```python
losses = collect_split_losses(
    model, validation_tokens, batch_size=4, block_size=8,
    eval_batches=5, seed=19,
)
summary = summarize_losses(losses)
```

All three routes give exactly the same sequence of losses for the
same model and seed. The helper calls `model.eval()`, disables
autograd, and restores the original training mode afterward.
It never advances the global CPU random generator.

The mean is still the average of the batch losses; dispersion now
adds detail without changing the objective or optimizer.

```bash
python -m pytest tests/test_evaluation_samples.py -q
```

**Experiment:** Print each batch loss before and after a small model
update, using the same sampled validation examples.

**Checkpoint:** Why is a tuple of per-batch losses more informative
than one scalar when batches vary in difficulty?
