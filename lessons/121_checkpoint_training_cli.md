# Lesson 121 — Enable memory-saving recomputation in the training CLI

Our transformer checkpointing code is now available from both:

```python
train_char_gpt(
    "data/tiny_story.txt", steps=2,
    attention_impl="sdpa", activation_checkpointing=True,
)
```

and

```bash
python -m rebuild_gpt.train_cli --steps 2 \
  --attention-impl sdpa --activation-checkpointing
```

This is a **training-time optimization**, not a new set of model
parameters, so our inference model format is unchanged. The running
trainer enables recomputation on the transformer stack, but disables
it automatically for evaluation and `no_grad` sections.

We record the flag in resumable training **controls when enabled**.
A restart cannot silently switch it off, while older checkpoints
created with the previous default still work.

```text
training step → checkpointed blocks → less saved activation state
validation    → ordinary blocks     → no recomputation needed
```

Run `python -m pytest tests/test_checkpointed_training.py -q`.

**Experiment:** Train with and without the flag using identical seeds.
Compare the weights after two optimizer updates.

**Checkpoint:** Why shouldn't a training-only checkpoint flag alter
the set of keys in the saved GPT weight state dictionary?
