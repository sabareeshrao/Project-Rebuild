# Lesson 85: opt in to nanoGPT's grouped AdamW in the text trainer

We now have two optimizer modes:

```python
train_char_gpt("data/tiny_story.txt")  # original tutorial behavior
train_char_gpt(
    "data/tiny_story.txt",
    grouped_adamw=True,
    weight_decay=0.1,
    adam_betas=(0.9, 0.95),
)
```

The grouped mode excludes biases and LayerNorm scale vectors from
weight decay, but decays matrix weights. It also configures AdamW
momentum coefficients explicitly. The helper records two optimizer
groups so the experiment is inspectable.

The default remains the old single-group AdamW path (PyTorch defaults,
including its default weight decay) to keep earlier seeded examples
numerically stable. Nondefault weight-decay/beta settings require
`grouped_adamw=True`, so options cannot be silently ignored.

Run: `python examples/82_train_grouped_adamw.py`.

**Experiment:** Compare the trainable weights after 10 updates with
`grouped_adamw=False` versus `True` using the same seed.

**Checkpoint:** Why can the same learning rate and dataset yield
different weights when the Adam betas and decay groups change?
