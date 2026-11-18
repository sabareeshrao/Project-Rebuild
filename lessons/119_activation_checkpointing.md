# Lesson 119 — Activation checkpointing: recompute instead of remember

A transformer needs intermediate activations during backpropagation.
With many layers, saving these can consume considerable memory.

**Activation checkpointing** trades compute for memory: we save less
information from the forward pass and recompute parts of each
transformer block as needed during backward.

```text
regular:       forward saves intermediates → backward reads them
checkpointed:  forward saves less           → backward recomputes
```

The stack now accepts `activation_checkpointing=True` or can enable
it at runtime using `set_activation_checkpointing(True)`. It wraps
each transformer block with PyTorch's non-reentrant checkpoint
(`use_reentrant=False`), preserving RNG state for dropout replay.

In `.eval()` mode and under `torch.no_grad()`, checkpointing is
disabled. It does not change parameter names, state-dict keys,
architecture dimensions, or checkpoint tensor shapes.

Run `python -m pytest tests/test_activation_checkpointing.py -q`.

**Experiment:** Compare the gradients of two identical stacks with
checkpointing switched on and off.

**Checkpoint:** Why can we save GPU memory while *increasing* training
time by recomputing block operations during backward?
