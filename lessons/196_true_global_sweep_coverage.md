# Lesson 196 — Report global coverage rather than summed seed coverage

The paired sample-budget sweep already included
`total_unique_corpus_positions_across_seed_plans`, but that value
is the **sum** of the individual seeds' distinct positions. It is
not the globally distinct token count.

If two seeds touch positions 0–19, the sum is 40, while the
global union still covers only 20 positions.

New report columns explicitly describe the distinction:

- `global_unique_corpus_token_positions`: true union of touched tokens.
- `global_corpus_coverage_fraction`: fraction of held-out corpus touched.
- `repeated_positions_between_seeds`: duplicated coverage across seeds.
- `duplicate_starts_between_seeds`: repeated exact start indices.

The original summed field is preserved for backwards compatibility.
Tests check that its difference from the union matches the new
between-seed repetition field.

```bash
python examples/110_paired_budget_sweep.py \
  --baseline out/before.pt --candidate out/after.pt \
  --corpus data/tiny_story.txt --sampling-policy all \
  --eval-batch-counts 1,2,4 --json out/coverage.json
```

Run `python -m pytest tests/test_cross_seed_budget_sweep.py -q`.

**Experiment:** Increase the seed count while keeping the corpus and
per-seed window count fixed.

**Checkpoint:** Why can summed coverage keep growing even after
the global union reaches the entire held-out corpus?
