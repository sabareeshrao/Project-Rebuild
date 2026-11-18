# Lesson 91: resume training from the command line

The training CLI now exposes `--training-state PATH` to save the
resumable optimizer checkpoint and `--resume-from PATH` to load one.
They are separate from `--output` (inference weights), `--best-output`
(best inference weights) and `--history-file` (JSON metrics).

```bash
# First session: run to optimizer update 2
python -m rebuild_gpt.train_cli --steps 2 --eval-every 2 \
  --training-state out/train_state.pt

# Second session: continue to TOTAL update 4 (not 4 extra updates)
python -m rebuild_gpt.train_cli --steps 4 --eval-every 2 \
  --resume-from out/train_state.pt --training-state out/train_state.pt
```

The CLI refuses to reuse an inference-output path as the state file.
It also records resume and output paths in the JSON report.

Run: `python -m pytest tests/test_cli_resume.py -q`.

**Checkpoint:** Why is a path to the best inference checkpoint not
sufficient for resuming AdamW training?
