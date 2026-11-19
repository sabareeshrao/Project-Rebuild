# Lesson 241: why one matching step is not enough

AdamW maintains moving averages of gradients and squared gradients.
If you recreate the optimizer on every step, you reset those
statistics and are **not** testing a real multi-step trajectory.

We now synchronize weights once, create both optimizers once, and
reuse them while repeating the same deterministic token batch.

```text
initialize model + optimizer
    | -> step 1 (AdamW moment 1) -> compare gradients/weights
    | -> step 2 (reuses moments) -> compare gradients/weights
    | -> step 3 (reuses moments) -> compare gradients/weights
```

We disable dropout via `eval()` during the experiment so no
random masks introduce a separate variable. Gradients are still
computed in eval mode. The original train/eval modes are restored.

```bash
python examples/134_grouped_training_trajectory.py \
  --upstream-model /trusted/nanoGPT/model.py --steps 3
```

**Experiment:** Set `--steps 1`, then `--steps 3`. The report should
contain exactly that many records.

**Checkpoint:** Why does this test offer stronger evidence than
three independent single-step comparisons, and why does it still
fall short of matching the full training loop?
