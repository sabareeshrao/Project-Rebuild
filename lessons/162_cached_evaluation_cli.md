# Lesson 162 — Control reproducible evaluation input preparation

The character-GPT CLI now exposes an opt-in evaluation backend:

```bash
python -m rebuild_gpt.train_cli \
  --steps 5 --eval-every 1 --eval-batches 4 \
  --validation-mode sampled --eval-batch-backend cached
```

The existing default `--eval-batch-backend regenerate` still
draws the same **seed-controlled** training/validation evaluation
batches at every check. The new `cached` backend draws those
batches only once, retains them on CPU, and evaluates them using
the current weights at every checkpoint.

The cache is not applicable to `--validation-mode full`, which
scores the entire contiguous token stream without sampling.

The selected mode is written to the JSON training report and, when
nondefault, to resumable training-state controls. A resumed run
cannot silently switch from cached to regenerated evaluation.
Older default-mode training states remain compatible.

Run `python -m pytest tests/test_cached_eval_cli.py -q`.

**Experiment:** Run three steps with the same seed using both
evaluation backends. Compare every recorded loss and final weights.

**Checkpoint:** If both backends use the same evaluation token IDs,
why might their wall-clock timings differ despite identical losses?
