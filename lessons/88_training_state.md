# Lesson 88: what must a resumable checkpoint remember?

A GPT inference checkpoint saves model weights and a tokenizer. Resuming
**training** also needs the AdamW moment estimates, exact global optimizer
step, PyTorch CPU RNG state (dropout), and batch-sampling generator state.
We also save validation history, the baseline loss and the best-model snapshot.

```text
model + tokenizer + optimizer + step + two RNG states
                  + corpus fingerprint + training controls
                         ↓
                  training-state .pt
```

The SHA-256 fingerprint checks **corpus bytes**, not the path. Different
text or line endings change the fingerprint. On load, we verify the model
configuration, token IDs, corpus fingerprint and training controls before
restoring anything.

Inference and training checkpoints intentionally have different formats.

**Experiment:** Run the new state-round-trip test and change a training
control. Verify that loading rejects the mismatch.

**Checkpoint:** Why do model weights alone fail to guarantee reproducible
continuation of AdamW training?
