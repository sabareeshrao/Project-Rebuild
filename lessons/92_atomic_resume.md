# Lesson 92: keep the last good checkpoint even if saving fails

A text-training state is useful only if it survives interruption.
Overwriting `training_state.pt` directly can leave it corrupted when a
process crashes or disk write fails.

We now write to a fresh temporary file **in the same directory** and
then replace the destination atomically using `os.replace()`. This
protects against a partially written new file becoming the published
checkpoint; it is not a substitute for backups or durable filesystem
sync on power loss.

```text
good state.pt ─── remains usable while ──> .state.pt.tmp is written
        └─────── atomic replace after success ───> new state.pt
```

Tests simulate `torch.save` raising `OSError`: the previous bytes must
remain unchanged and no temporary files remain.

Run:

```bash
python -m pytest tests/test_text_resume.py -q
python -m pytest tests/test_text_training_resume.py tests/test_cli_resume.py -q
python examples/85_resume_text_gpt.py
```

**Experiment:** Stop training at an evaluation boundary, resume it, and
compare with one uninterrupted run. Try changing `--seed` when resuming;
the checkpoint should reject mismatched training controls.

**Checkpoint:** Why is `os.replace` safer than opening the existing
checkpoint path in write mode?

**Limitations:** This lesson is CPU-only. It does not resume CUDA RNG or
mixed-precision gradient scaler state; those require additional work.
