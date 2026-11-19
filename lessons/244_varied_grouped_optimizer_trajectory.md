# Lesson 244 — Compare optimizer updates on changing inputs

A realistic optimizer's gradient changes for two reasons: the weights
have changed, and the next training batch contains different tokens.
To isolate **implementation parity**, both GPTs must receive the exact
same step-by-step batch sequence.

This lesson adds a backwards-compatible `batches` argument to the
grouped AdamW trajectory comparer. When omitted, the old fixed-batch
experiment remains available.

```text
train split -> seed -> batch 1 -> both models -> compare
                   -> batch 2 -> both models -> compare
                   -> batch 3 -> both models -> compare
```

The original source is opt-in and must be independently reviewed.

```bash
python examples/120_capstone_cli.py
python examples/137_varied_grouped_trajectory.py \
    --upstream-model /trusted/nanoGPT/model.py
```

**Checkpoint:** What would happen if each GPT drew its own random
batches rather than sharing an identical deterministic sequence?
