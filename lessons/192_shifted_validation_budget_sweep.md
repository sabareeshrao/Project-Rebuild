# Lesson 192 — Compare fixed and shifted disjoint tiling fairly

Our paired saved-checkpoint budget sweep now supports all three
sampling policies: `replacement`, `disjoint`, and `shifted`.
Choose `--sampling-policy all` to compare every policy using the
same seed count, batch size, context, and evaluation budgets:

```bash
python examples/110_paired_budget_sweep.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt --sampling-policy all \
  --eval-batch-counts 1,2,4 --seed-count 3 \
  --csv out/shifted_sweep.csv --json out/shifted_sweep.json
```

Every row records a signed paired mean delta, descriptive spread,
window overlap, and chosen lattice phases by seed (or `n/a` for
replacement). Equal sampled target counts do **not** mean equal
sampling distributions.

**Subtle reproducibility limit:** the original and fixed-origin
disjoint policies reuse sampled prefixes when the budget grows.
For shifted tiling, near-capacity requests restrict feasible phases;
a larger budget can change the chosen phase. Therefore the
corresponding rows need not share sample prefixes even for the
same seed. The JSON settings explicitly disclose that distinction.

The experiment cannot prove which sampling distribution produces
the best held-out estimate. All windows still come from one corpus,
and seed-level observations may be dependent.

Run `python -m pytest tests/test_shifted_budget_sweep.py -q`.

**Experiment:** Compare the same checkpoint against itself under all
policies. Paired deltas should all be zero despite differing phases.

**Checkpoint:** Why must the budget-sweep report distinguish nested
samples from samples whose feasible start grid changes with budget?
