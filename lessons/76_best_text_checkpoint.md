# Lesson 76: save the best text model, not automatically the last

We can now use the validation estimates collected during training to
choose the checkpoint with the **lowest observed validation loss**.

Set `eval_interval` and optionally:

```python
train_char_gpt(
    corpus_path,
    eval_interval=10,
    restore_best=True,
    best_checkpoint_path="best.pt",
)
```

At step 0, each evaluation interval and the final step, the trainer
compares loss on the **same seeded held-out batches**. When a new best
score appears it clones that model's weights, and optionally saves a
self-contained checkpoint containing the tokenizer too.

If `restore_best=True`, the returned model is restored to that best
snapshot. The `validation_after` metric is computed after restoration.
If `restore_best=False`, the returned model remains at the last
training step, while the best checkpoint still exists on disk.

Run: `python examples/73_restore_best_text_gpt.py`.

**Experiment:** Compare the final training step and `best_step`
on a small corpus. Why might they differ?

**Checkpoint:** Why must selecting a checkpoint use **validation**
loss rather than the last training mini-batch's loss?

**Caveat:** A few fixed minibatches provide a noisy proxy for true
held-out performance, and repeated checkpoint selection can overfit
that fixed validation sample. More thorough evaluation comes later.
