# Lesson 249 — Compare every byte of meaningful training state

Comparing final GPT weights is only half of a resume-parity check.
Two checkpoints can share weights but disagree in AdamW momentum or
the RNG state that determines future batches.

Our `compare_resume_checkpoints` requires exact equality across
model tensors, optimizer groups and moments, CPU RNG, batch sampler
RNG, training step, configuration and progress metadata.

```text
full run state      resumed run state
  model weights       model weights
  AdamW moments       AdamW moments
  RNG streams         RNG streams
  controls/history    controls/history
          \           /
           exact comparison
```

This is intentionally **bitwise equality** on one CPU/PyTorch
environment, not approximate tolerances for mixed-precision GPU runs.

After the next lesson creates both checkpoints, run
`python examples/142_compare_resume_states.py`.

**Question:** What can go wrong on the next optimizer step even if
all model weights currently match perfectly?
