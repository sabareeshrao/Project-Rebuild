# Lesson 166 — Set a cache-memory budget from the training CLI

When a run uses `--eval-batch-backend cached`, it can now enforce a
conservative CPU memory ceiling:

```bash
python -m rebuild_gpt.train_cli --steps 5 --eval-every 1 \
  --eval-batch-backend cached --eval-cache-max-mib 16
```

The CLI accepts a positive MiB value, estimates the retained int64
input and target tensor elements across **both** evaluation splits,
and rejects the run if the estimated logical bytes exceed that cap.

The JSON training report contains the requested cap and estimated
logical bytes. This is a **budget bound**, not measured GPU memory
or total process RSS. The cache always holds its token batches on CPU.

A cap without cached sampled evaluation is rejected rather than
ignored. It is also preserved in resumable training controls, so
accidentally changing the cap across a restart produces a setup
mismatch error.

The default behavior is unchanged: no cap and regenerated evaluation
batches unless explicitly requested.

Run `python -m pytest tests/test_eval_cache_budget_cli.py -q`.

**Experiment:** Run with a cap that fits, then lower it below the
reported logical bytes. Check that the CLI fails before writing model
or history outputs.

**Checkpoint:** Why is an explicit user-provided budget preferable
to silently changing the number of validation samples?
