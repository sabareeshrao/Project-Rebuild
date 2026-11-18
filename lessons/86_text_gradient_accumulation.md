# Lesson 86: train GPT using multiple microbatches per optimizer step

The text trainer can now accumulate gradients over K separate
mini-batches and update weights **once** per outer iteration.

```text
for each optimizer step:
   sample K independent batches [B,T] from train split
   zero gradients
   backward on each batch with loss divided by K
   clip accumulated gradients once, if enabled
   perform one AdamW update
   evaluate/checkpoint as before
```

The effective amount of training data in an update is
`batch_size * block_size * gradient_accumulation_steps` token
positions (with sampling replacement). The learning-rate schedule
counts **optimizer updates**, not microbatches.

```python
train_char_gpt(
    "data/tiny_story.txt",
    gradient_accumulation_steps=3,
    grouped_adamw=True,
    max_grad_norm=1.0,
)
```

With accumulation set to 1 (the default), the trainer uses its
original single-step helper; earlier experiments retain identical
weights and metrics.

Run: `python examples/83_accumulated_text_training.py`.

**Experiment:** Hold outer optimizer steps fixed and change
accumulation from 1 to 4. How many input tokens are sampled?

**Checkpoint:** Why does the scheduler advance once for K
microbatches instead of advancing K times?
