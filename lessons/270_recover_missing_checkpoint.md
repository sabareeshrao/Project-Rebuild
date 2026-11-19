# Lesson 270: recover from a missing checkpoint, not just modified bytes

A checkpoint may be unavailable because of an interrupted upload,
accidental deletion, or storage failure. With a trusted multistage
audit, we can skip it and choose a previously verified checkpoint.

Run a complete missing-checkpoint drill **against disposable copies**:

```bash
python examples/150_multistage_resume_audit.py
python examples/163_missing_checkpoint_recovery.py
```

The drill does three critical checks: the missing file is not loaded,
the latest earlier intact stage is chosen, and the replayed model,
AdamW moments, and RNG state match the uninterrupted reference exactly.

You can also use the existing command with `--fault-mode remove`
or `--fault-mode truncate`. By default it retains the original
`append` behavior.

**Checkpoint:** Why is training from step zero unnecessary when
a prior optimizer checkpoint includes the correct RNG and moments?
