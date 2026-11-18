# Lesson 47: train and validate a deeper causal GPT

We can now train a model with **more than one transformer block** using
our existing AdamW training helper. To isolate the effect of context,
we repeat the same two next-token tasks from earlier lessons:

```text
input [0,1] -> target [1,0]
input [2,1] -> target [1,2]
```

At index 1, the **current token** and absolute **position** are identical.
The model must use the first token to choose the second target.
A two-block model can learn this via causal self-attention.

We also test **prefix invariance** in Lesson 45: changing input tokens
after a prefix must not change predictions for that prefix, even when
information passes through multiple attention layers.

Run: `python examples/47_train_stacked_gpt.py`.

**Experiment:** Repeat training with `n_layer=1` and `n_layer=3`.
Use the same seed and compare trainability, parameter counts, and
training speed. More depth is not automatically better on tiny data.

**Checkpoint:** Identify the operations that share parameters across
positions, and explain why increasing model depth doesn't allow
the network to see future tokens.

**What's next:** Refine model wiring toward original nanoGPT (including
tied token/output weights and GPT-style initialization), then build
evaluation, sampling, and checkpoint workflows. This educational model
is functional but is not yet a checkpoint-compatible nanoGPT clone.
