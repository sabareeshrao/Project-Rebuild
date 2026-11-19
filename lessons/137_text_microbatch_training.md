# Lesson 137 — Train real GPT with an exact logical batch

Our text trainer now has an optional `microbatch_size` that splits
**one logical batch** sampled from the corpus into smaller chunks.
The default remains unchanged.

```bash
python -m rebuild_gpt.train_cli \
  --steps 2 --batch-size 5 --microbatch-size 2
```

This example samples **five sequences per optimizer update**, with
three forward/backward passes of sizes (2,2,1). The target loss is
weighted by its number of token positions, giving every token the
same contribution. The optimizer updates **once**.

Contrast two independent ways of using small batches:

```text
--batch-size 5 --microbatch-size 2
    → one logical B=5 batch, split (2,2,1), 5×T tokens/update

--batch-size 2 --gradient-accumulation-steps 3
    → three independent B=2 batches, 6×T tokens/update
```

To avoid ambiguity, this educational CLI rejects combining the new
`--microbatch-size` option with accumulation steps greater than 1.

The selected microbatch size is recorded in resumable training-state
controls **only when the new mode is active**. Older checkpoints made
with the default remain compatible, but resuming with a different
splitting policy is rejected. Training-history JSON also reports
the microbatch size and effective tokens/update.

Run `python -m pytest tests/test_text_microbatch_cli.py -q`.

**Experiment:** Train with B=5 and M=5,2,1 from the same seed.
Compare loss and saved parameter tensors.

**Checkpoint:** Why does M=2 use more forward/backward passes than
M=5 even though both steps see the same five training examples?
