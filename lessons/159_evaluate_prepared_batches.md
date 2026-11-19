# Lesson 159 — Reuse examples but still evaluate new model weights

Reusing sampled validation batches is useful only if we still run
each **current model** on them at every evaluation step.

The new `evaluate_prepared_split_loss` does exactly this: it moves
the existing CPU batches to the model device and runs forward passes
under `eval()` and `no_grad()`, restoring the original train/eval
mode even if evaluation fails.

```text
precomputed validation token IDs (unchanged)
              |
        current model at step N → loss N
        current model at step M → loss M
```

The train and validation plans use independent seeds, as in the
original evaluator. Tests compare their losses *exactly* against
the legacy regenerate-every-time approach on CPU and confirm that
model parameters and global CPU RNG state are not changed.

Cached input preparation does not cache model logits or losses.
It also does **not** pin or pretransfer all samples to CUDA, so
device transfer costs remain part of every evaluation call.

Run `python -m pytest tests/test_prepared_evaluation.py -q`.

**Experiment:** Change some weights and run evaluation again using
the same plan. Why should the loss change?

**Checkpoint:** Why do evaluation helpers use both `model.eval()`
and `torch.no_grad()` rather than merely reusing precomputed inputs?
