# Lesson 199 — Plot *new* validation coverage per seed

The model-free seed-coverage sweep now exports a per-seed
`marginal_coverage` trace in JSON, alongside its existing
per-configuration summary rows.

For each ordered seed, the trace records:

- Token positions it adds to the global corpus union.
- Positions already touched by earlier seed plans.
- Repeated positions inside its own sampled windows.
- The running cumulative unique coverage and coverage fraction.

```bash
python examples/111_seed_coverage_sweep.py \
  --corpus-tokens 1024 --seed-counts 1,2,4 \
  --sampling-policy all --json out/marginal_coverage.json
```

CSV gains `latest_seed_new_positions` and
`zero_new_coverage_seeds`, without removing earlier columns.
For a fixed policy, the final global union equals the sum of
all per-seed newly covered positions. The detailed JSON trace
makes this identity easy to verify.

The sequence is **order-dependent**: earlier seeds claim positions
before later seeds can add them. This is not a measure of model
quality, effective sample size, or statistically independent
observations.

Run `python -m pytest tests/test_marginal_coverage_report.py -q`.

**Experiment:** Which policy adds the fewest new positions with
the fourth seed? Does the answer change with corpus size?

**Checkpoint:** Why can a zero-coverage-gain seed still have
a different sampled validation *loss* than earlier seeds?
