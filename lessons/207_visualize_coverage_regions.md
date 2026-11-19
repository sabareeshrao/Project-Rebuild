# Lesson 207 — Export plot-ready regional validation coverage

The held-out coverage example can now save a **long-format CSV**
with one row per corpus region per sampling-policy/seed-count
combination, instead of requiring another script to flatten JSON.

```bash
python examples/112_heldout_seed_coverage.py \
  --corpus data/tiny_story.txt --position-bins 8 \
  --seed-counts 1,2,4 --sampling-policy all \
  --csv out/coverage_summary.csv \
  --position-bin-csv out/coverage_regions.csv \
  --json out/coverage_details.json
```

Each positional row contains the sampling policy, seed count,
region index, half-open token boundaries, distinct positions
touched, total visits, repeated visits, and covered fraction.

You can plot `coverage_fraction` against `bin_index`,
coloring by sampling policy or seed count. This reveals blind
spots that a single global coverage percentage may hide.

The export is opt-in, refuses paths that could overwrite the
original corpus or another output, and preserves existing CSV/JSON
fields when omitted. These metrics are not independent sample
counts or calibrated measures of validation quality.

Run `python -m pytest tests/test_heldout_position_csv.py -q`.

**Experiment:** Plot disjoint versus replacement sampling with
the same seed count and context. Which regions are unsampled?

**Checkpoint:** Why is a row per corpus region easier to chart
than a single global coverage statistic?
