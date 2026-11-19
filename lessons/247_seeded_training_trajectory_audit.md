# Lesson 247 — Audit a reproducible changing-batch training trajectory

The comparison now supports two explicit modes:

- `fixed_validation` (earlier experiment): repeat a held-out batch
  strictly to inspect optimizer behavior.
- `train_sequence` (new): draw a **new training batch each step** and
  compare both GPTs using the same deterministic batch sequence.

In either mode the optimizer state persists and each step now checks
the gradients, model tensors and AdamW internal moments.

```text
train text -> token IDs -> seeded batches (one per step)
                           |               |
                          our GPT       nanoGPT
                            |              |
                    same AdamW setup and learning signal
                            |              |
                    check moments + updated parameters
                            |              |
                   JSON audit if everything agrees
```

Run with a reviewed original source (never downloaded automatically):

```bash
python examples/120_capstone_cli.py --steps 4
python examples/140_training_sequence_audit.py \
  --upstream-model /trusted/nanoGPT/model.py \
  --upstream-sha256 <verified-sha256> --steps 3
```

**Question:** Which major parts of the upstream training system have
we still NOT reproduced? Consider distributed sampling, training-mode
dropout, optimizer schedules, mixed precision, and longer training runs.
