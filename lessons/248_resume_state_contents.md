# Lesson 248 — What must a training checkpoint remember?

An inference checkpoint only needs weights and the tokenizer. A **training
resume checkpoint** also needs optimizer momentum, step counters and
random-stream positions to continue the same training trajectory.

```text
model weights + optimizer state + training step
              + CPU RNG + batch-sampler RNG + corpus identity
              = reproducible resume ingredients
```

We inspect our pre-existing `text_resume.py` training-state format with
`inspect_resume_checkpoint`; we don't invent a second checkpoint format.

After producing a checkpoint in the next lesson, run:
`python examples/141_inspect_resume_state.py out/resume_lab/partial.pt`.

**Experiment:** Why would restoring model weights **without** AdamW's
`exp_avg` and `exp_avg_sq` change the next optimizer update?

**Checkpoint:** Why is saving the training-batch RNG separate from
saving the model's global CPU RNG state?
