# Lesson 73: evaluate without changing the experiment

A single randomly sampled mini-batch is a noisy measurement of model
quality. We now estimate **train and validation cross-entropy** by
averaging several mini-batches, generated from fixed seeds.

```text
unchanged model -> eval mode -> seeded train batches -> mean train loss
                -> eval mode -> seeded val batches   -> mean val loss
```

Evaluation uses `torch.no_grad()` and temporarily disables dropout.
It restores the caller's training/evaluation mode afterward. A
dedicated `torch.Generator` is used for sampling, so evaluating does
not consume the process-wide RNG or change the next training batch.

Run: `python examples/70_evaluate_splits.py`.

**Experiment:** Increase `eval_batches` from 1 to 20. How much does
the estimated loss vary between different seeds?

**Checkpoint:** Why should we use the *same* evaluation samples when
comparing checkpoints, and why doesn't this estimate equal the exact
loss over all possible corpus windows?

**Scope:** These are reproducible sampled estimates, not a full-corpus
evaluation or a held-out test-set score.
