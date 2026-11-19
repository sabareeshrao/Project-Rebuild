# Lesson 211 — Record start balance in saved GPT comparisons

The saved-model comparison CLI now supports an explicit
region-balanced validation policy:

```bash
python examples/109_compare_saved_checkpoints.py \
  --baseline out/before.pt --candidate out/after.pt \
  --corpus data/tiny_story.txt \
  --sampling-policy balanced --sampling-regions 5 \
  --eval-batches 4 --csv out/balanced.csv \
  --json out/balanced.json
```

CSV reports regional start counts, empty start regions,
maximum/minimum imbalance and the paired loss for each seed.
JSON preserves the region boundaries and counts, along with
existing overlap and cross-seed coverage metadata.

The default remains uniform start sampling with replacement.
The balanced policy retains no nonoverlapping tile lattice;
the lattice phase is reported as null. Unknown/invalid region
settings fail rather than being silently ignored.

This distribution is deliberately different from the original
start sampler. Balanced *starts* are not uniformly distributed
token-context windows and do not guarantee complete regional
token coverage. Model comparisons remain descriptive.

Run `python -m pytest tests/test_balanced_checkpoint_cli.py -q`.

**Experiment:** Compare two identical checkpoints and vary the
number of sampling regions. Which metadata changes, and why
must paired model deltas remain exactly zero?

**Checkpoint:** Why is storing the selected policy and region
boundaries necessary to interpret saved evaluation reports?
