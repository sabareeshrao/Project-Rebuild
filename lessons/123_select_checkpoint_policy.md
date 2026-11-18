# Lesson 123 — Choose *which* transformer blocks to recompute

Previous lessons checkpointed either **all blocks** or **no blocks**.
Now we define a third choice: checkpoint *every Nth block*.

The helper `checkpointed_indices(n_layer, every_n)` returns zero-based
positions. For four blocks:

```text
n_layer = 4
every_n = None   → ()           → none
every_n = 1      → (0, 1, 2, 3) → all
every_n = 2      → (1, 3)       → every second
every_n = 4      → (3,)         → last only
```

Layer counting in the policy is human-friendly (1, 2, ...), while
Python indexing is zero-based (0, 1, ...). Incomplete final groups
are not checkpointed.

This commit only selects indices. It does **not** alter the model or
run an experiment yet. Keeping the policy pure allows us to verify
its meaning before changing the forward path.

Run: `python -m pytest tests/test_checkpoint_policy.py -q`.

**Experiment:** For seven blocks, list the indices selected by
`every_n=3`.

**Checkpoint:** Why should the policy refuse `every_n=0`
rather than silently treating it like no checkpointing?
