# Lesson 81: put clipping and learning-rate decay into training

The previous two lessons built independently testable primitives.
Now the character-GPT training loop can use them together, without
altering its earlier defaults.

Enable the new controls with:

```python
train_char_gpt(
    "data/tiny_story.txt",
    max_grad_norm=1.0,
    warmup_steps=3,
    lr_decay_steps=25,
    min_lr=0.002,
)
```

At each **zero-based optimizer iteration**, the schedule calculates
an LR; the trainer applies it to each AdamW parameter group.
After backpropagation, optional gradient clipping caps the global
norm before calling `optimizer.step()`.

Without `lr_decay_steps`, the learning rate remains constant and
`warmup_steps` / `min_lr` must be zero. Without `max_grad_norm`,
gradients are not clipped. Earlier exercises therefore retain
their previous numerical results.

Run: `python examples/78_train_with_schedule.py`.

**Experiment:** Compare models trained with and without these controls
using the same seed and corpus. Do their final weights differ?

**Checkpoint:** Why is the scheduler evaluated *before* each optimizer
step, and why does a learning-rate schedule not necessarily improve a
tiny corpus's validation loss?
