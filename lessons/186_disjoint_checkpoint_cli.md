# Lesson 186 — Disclose the validation sampling policy in reports

Saved-model comparisons now have an explicit CLI choice:

```bash
python examples/109_compare_saved_checkpoints.py \
  --baseline out/before.pt --candidate out/after.pt \
  --corpus data/tiny_story.txt \
  --sampling-policy disjoint --eval-batches 3 \
  --csv out/disjoint.csv --json out/disjoint.json
```

The default is still `replacement`: independently drawn random
start positions, potentially including repeated or overlapping
token windows.

`disjoint` chooses a seeded permutation of fixed nonoverlapping
T+1-token tiles. If the held-out corpus has too few distinct tiles,
the CLI fails instead of quietly sampling the same tile twice.

CSV records the policy, duplicate starts, repeated window-token
positions and unique covered token positions per seed. JSON includes
these overlap diagnostics plus the exact sample configuration,
paired deltas and completed improvement/worsening/tie totals.

**Important:** The disjoint policy samples a different distribution
and does not eliminate correlations between windows from one
corpus. These reports are not statistical significance tests.

Run `python -m pytest tests/test_disjoint_checkpoint_cli.py -q`.

**Experiment:** Compare one saved checkpoint with itself using
both policies. Which metrics must stay exactly equal?

**Checkpoint:** Why is publishing the sampling policy alongside
the loss delta necessary for reproducibility?
