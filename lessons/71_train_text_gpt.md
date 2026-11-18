# Lesson 71: train on text, not handcrafted token IDs

Our data pipeline already reads UTF-8 text, creates a deterministic
character tokenizer, splits the token stream, and samples batches
of next-token input/target pairs. We now connect it to GPT training:

```text
tiny_story.txt -> character IDs -> contiguous train/validation split
                -> random [B,T] input/target batches
                -> GPT + cross-entropy + AdamW
                -> save trained model + tokenizer in one checkpoint
```

We evaluate **the same seeded validation batch** before and after
training. The validation split is contiguous and not used for optimizer
updates. The vocabulary is built from the full corpus in this simple
exercise; more rigorous datasets would fix the tokenizer independently.

Run:

```bash
python examples/68_train_text_gpt.py --steps 60 --output tiny_story_gpt.pt
```

This training loop uses a tiny model and corpus for demonstration.
A lower validation loss is not guaranteed on every dataset, and
the generated text may still be incoherent. The saved bundle is for
**inference**; resuming AdamW requires our separate training-checkpoint
workflow.

**Experiment:** Compare 10 and 100 steps. How do training and
validation loss differ?

**Checkpoint:** Why should we measure validation loss separately
from the loss on each randomly sampled training batch?
