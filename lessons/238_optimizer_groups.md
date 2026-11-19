# Lesson 238: why nanoGPT has two AdamW parameter groups

AdamW applies **decoupled weight decay** to matrix-shaped learnable
weights but excludes rank-1 biases and LayerNorm scale parameters.
Our earlier `configure_adamw` already implements this rule;
today we make the rule explicit and inspectable.

```text
model.named_parameters()
  ├─ ndim >= 2 -> decay (e.g. attention matrices, embeddings)
  └─ ndim <  2 -> no_decay (e.g. biases, LayerNorm scales)
```

The token embedding and output head may share one underlying
parameter. Iterating unique parameters means the shared weight must
not be updated twice.

Run `python examples/131_optimizer_groups.py`.

**Experiment:** Compare grouped tensor counts for one and three GPT
blocks; why is the shared embedding still counted only once?

**Checkpoint:** Weight decay is distinct from a gradient penalty.
Which tensor ranks receive it in the original nanoGPT convention?
