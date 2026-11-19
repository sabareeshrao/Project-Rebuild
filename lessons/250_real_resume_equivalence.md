# Lesson 250 — Stop training, reload, and finish without changing the result

A stronger test of our nanoGPT rebuild is to run the **same training
experiment two ways**:

```text
                    4 steps -> full.pt
start weights /
              \\ 2 steps -> partial.pt -> load -> 2 more -> resumed.pt
                                                        |
                    compare everything at final step --+
```

Both runs use the same seed, mini-batch size, grouped AdamW optimizer,
cosine learning-rate schedule, and CPU model. We evaluate every step
so the early checkpoint and the uninterrupted trajectory follow the
same evaluation cadence.

The existing trainer already supports saving and resuming state.
We deliberately reuse it instead of writing a second training loop.

Run `python examples/143_resume_equivalence.py`. This creates three
checkpoints in `out/resume_lab`.

**Experiment:** If you restore only weights but omit the optimizer
moment and batch RNG state, can you expect bit-for-bit agreement?
Why not?

**Checkpoint:** Equal last-step validation losses alone are not enough:
which extra checkpoint fields must agree to claim resume parity?
