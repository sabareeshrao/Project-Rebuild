# Lesson 53: read a model's state dictionary

A PyTorch `state_dict` is a mapping from **tensor names** to tensor
values. It includes learnable parameters and persistent buffers.
Its keys and shapes form the interface that checkpoints depend on.

Our fused GPT currently uses names such as:

```text
tokens.lookup.weight                      [V,C]
positions.lookup.weight                   [T,C]
blocks.blocks.0.attention.qkv.c_attn.weight [3C,C]
blocks.blocks.0.mlp.expand.weight         [4C,C]
head.linear.weight                        [V,C]
```

Tied `tokens.lookup.weight` and `head.linear.weight` appear as
two *names* in `state_dict()`, but they reference one underlying
Parameter. `named_parameters()` normally deduplicates it.

Run: `python examples/53_inspect_state.py`.

**Experiment:** Compare the report with `tie_embeddings=False`.
Which aliases disappear and how does the number of unique weights change?

**Checkpoint:** Why does matching names without checking tensor
shapes risk loading a checkpoint incorrectly?

This commits only an inspection tool; it does **not** alter weights.
