# Lesson 06: measure generalization with a held-out split

If we only measure performance on text the model trained on, it may look
good because it memorized that data. A held-out **validation** segment
helps us estimate how well training generalizes to unseen text.

For a 10-token stream with `val_fraction=0.2`:

```text
[ 0 1 2 3 4 5 6 7 ] [ 8 9 ]
       train            validation
```

We preserve original order. Randomly scrambling individual tokens would
destroy the next-token prediction task. Both sets require at least two
tokens to form an input-and-target pair.

Run: `python examples/06_split_train_val.py`

**Experiment:** Change the validation fraction to 0.3; observe the sizes.

**Checkpoint:** Why should we never include validation targets in batches
used to update model parameters?

**Practical note:** This toy tokenizer builds its character vocabulary
from the complete file. Later we will discuss tokenizer fitting and
evaluation leakage more carefully for realistic datasets.
