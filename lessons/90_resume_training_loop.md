# Lesson 90: resume the same training trajectory

A resumable checkpoint is useful only when the training loop actually
restores it. `train_char_gpt(..., resume_from=path)` now checks that the
corpus, vocabulary, architecture and training controls match, then restores:

- current model weights (not best-validation inference weights)
- AdamW moments and per-group learning rates
- next optimizer-step number
- global CPU RNG for dropout and separate batch-sampling RNG
- prior validation history and best-model snapshot

`steps` means the **total number of optimizer updates**, not additional
updates: first run to step 2, then resume with `steps=4` for two more.

The scheduler advances by global optimizer step, so it does not restart.
For comparable evaluation histories, stop at an evaluation boundary:
the trainer also evaluates at the final step of each invocation.

Run: `python -m pytest tests/test_text_training_resume.py -q`

**Experiment:** Train for four steps without interruption and compare
its tensors with a run split into two plus two steps.

**Checkpoint:** Why does a checkpoint made *after* restoring the
best-validation model fail to resume the original optimizer trajectory?
