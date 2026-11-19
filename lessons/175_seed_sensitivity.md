# Lesson 175 — Compare validation means across sampling seeds

The current sampled evaluator deliberately uses one fixed seed so
loss curves are comparable between training steps. But that single
sample of windows may be unusually easy or unusually difficult.

`evaluate_seed_sensitivity` holds model weights fixed while running
the same evaluation procedure with a **distinct, preselected set of
seeds**:

```text
fixed model
   ├─ seed 10 → batch means → validation mean 10
   ├─ seed 12 → batch means → validation mean 12
   └─ seed 15 → batch means → validation mean 15
                           ↓
               summarize variation across seeds
```

The API records per-seed means, their descriptive spread and the
number of batches evaluated per seed. It rejects duplicated or
invalid seeds, leaves parameters and RNG untouched, and restores the
model's prior train/eval mode.

**Statistical caution:** changing the sampling seed is not the same
as obtaining a new independent held-out dataset. These samples
still come from one corpus, often with overlapping windows. This
is sensitivity analysis, not a calibrated confidence interval.

Run `python -m pytest tests/test_seed_sensitivity.py -q`.

**Experiment:** Increase the number of batches per seed. Does the
variation between seed means get smaller on your corpus?

**Checkpoint:** Why can multiple seeds improve our understanding
of sampling sensitivity without increasing the amount of genuinely
independent validation data?
