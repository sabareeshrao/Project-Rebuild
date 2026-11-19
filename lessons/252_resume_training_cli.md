# Lesson 252 — One command to verify training interruption recovery

The last five lessons study a central requirement of real model
training: stopping a run should not silently change the optimizer's
behavior when training resumes later.

Our command drives the existing trainer twice, with identical settings:

```text
uninterrupted (4 steps) -> full.pt
interrupted (2 steps)   -> partial.pt -> resume to 4 -> resumed.pt
                                                  |
                   exact model + AdamW + RNG + history check
                                                  |
                                  resume_audit.json (passed only if equal)
```

Run from the repository root:

```bash
python examples/145_resume_audit_cli.py \
  --corpus data/tiny_story.txt \
  --output-dir out/resume_lab \
  --first-steps 2 --total-steps 4 --seed 13
```

Inspect both checkpoint content and the resulting JSON report.

**Experiment:** Save a copy of the full checkpoint and intentionally
alter an AdamW moment, then compare the two saved states. Which
difference is reported?

**Checkpoint:** Does exact restart equality for a small CPU experiment
prove equivalent GPU/distributed training or original nanoGPT
training-loop parity? Explain the remaining differences.
