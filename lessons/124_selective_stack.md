# Lesson 124 — Recompute selected transformer blocks only

The stack now exposes `set_checkpoint_interval(N)`, using our pure
policy helper to pick every Nth block during gradient-enabled training.

```python
stack.set_checkpoint_interval(2)
assert stack.checkpointed_blocks == (1, 3)  # four-block model
```

The previous `set_activation_checkpointing(True)` still checkpoints
**all** blocks; setting it to False checkpoints **none**. Switching
between the old boolean and the new interval clears the other policy.

Only selected blocks use `torch.utils.checkpoint.checkpoint` with
`use_reentrant=False` and `preserve_rng_state=True`. Evaluation and
`no_grad` forward passes use ordinary block execution.

Nothing in the transformer parameters or `state_dict` changes.

Run: `python -m pytest tests/test_selective_stack.py -q`.

**Experiment:** Select every third block of a four-block model.
Which layer is actually recomputed?

**Checkpoint:** Why does a checkpoint policy have to be applied only
when gradients are needed, not during inference?
