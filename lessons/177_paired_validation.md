# Lesson 177 — Compare two GPTs on exactly the same sampled windows

When comparing model weights before and after training, using
different validation windows creates unnecessary sampling noise.
A **paired** comparison evaluates both models on the same fixed
windows for each seed and then compares their losses.

```text
validation seed 8 → same windows → baseline GPT loss
                                → updated GPT loss
                                         |
                          delta = updated - baseline
```

A negative delta means the updated GPT scored lower loss on those
windows. A positive delta means higher loss. Neither proves the
model generalizes better or worse on new data.

`compare_paired_validation` creates one index plan per seed, reuses
it for both models, and reports individual seed means and paired
differences. It preserves train/eval modes, model parameters and
global RNG state. The delta spread is **descriptive**, not a
statistical significance test.

The example makes one genuine AdamW update, then compares before
and after on a held-out text split:

```bash
python examples/108_paired_validation.py \
  --corpus data/tiny_story.txt --seed-count 3 \
  --csv out/paired.csv --json out/paired.json
```

Run `python -m pytest tests/test_paired_evaluation.py \
tests/test_paired_validation_cli.py -q`.

**Experiment:** Repeat with a different model seed. Does one update
always improve the held-out loss? Why not?

**Checkpoint:** Why does pairing examples remove one source of
comparison noise without making these held-out draws independent?
