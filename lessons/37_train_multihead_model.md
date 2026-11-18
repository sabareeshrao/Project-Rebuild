# Lesson 37: prove multi-head attention learns from earlier context

Our first 35 commits created a single-head context-sensitive model.
Now we use **two independently learned heads** on the same tiny task.

```text
inputs       [0, 1]       [2, 1]
targets      [1, 0]       [1, 2]
                  ^            ^
       same current token and absolute position,
       different correct answer based on the preceding token
```

A bigram or token-plus-position-only model cannot give different
predictions at position 1 here. Multi-head causal attention *can*
use the earlier token. We train for 120 AdamW steps and verify the
model predicts both target sequences.

Run: `python examples/37_train_multihead_model.py`

**Experiment:** Compare training with `n_head=1` and `n_head=2`,
keeping `n_embd=8` and the same seed. Both may solve this simple
task: head count alone does not guarantee better performance.

**Checkpoint:** Why can the prediction for the second token depend
on the first token, but not on a future third token?

**Next architectural milestones:** attention-probability dropout,
LayerNorm, feed-forward networks and stacked transformer blocks. Our
multi-head implementation uses independent heads for readability;
later we will study fused QKV projections for efficiency.
