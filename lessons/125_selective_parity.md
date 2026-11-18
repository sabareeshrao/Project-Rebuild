# Lesson 125 — Prove selective activation checkpointing is equivalent

Selecting fewer checkpointed blocks changes **where recomputation
happens**, not the language-model objective or parameter structure.

To verify that claim, we construct two GPTs with identical weights
and compare them under both manual-fused and SDPA attention:

- Regular forward/backward with dropout enabled.
- Selective checkpointing every 1st, 2nd, 3rd or 4th block.
- Loss values and every trainable parameter's gradient.
- CPU random-number generator state after backward.
- One AdamW parameter update from an identical starting state.

```text
same model, same seed, same tokens
              |
       uncheckpointed   vs   selected blocks
              |                    |
       loss → backward       loss → recomputation → backward
              |                    |
             compare loss, gradients, RNG and AdamW results
```

PyTorch's non-reentrant checkpoint with `preserve_rng_state=True`
is important when dropout is enabled. On CPU, the comparisons should
agree within small floating-point tolerances. Exact CUDA parity also
depends on kernels and hardware, and requires a GPU test environment.

Run `python -m pytest tests/test_selective_checkpoint_parity.py -q`.

**Experiment:** Turn off `preserve_rng_state` in a disposable branch.
Does the dropout-gradient parity test still pass?

**Checkpoint:** Why isn't verifying the output shape sufficient to
justify a training optimization?
