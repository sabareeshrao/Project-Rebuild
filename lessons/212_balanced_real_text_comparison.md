# Lesson 212 — Compare four sampling policies on held-out text

The held-out coverage CLI adds opt-in `balanced` starts and an
`extended` sweep comparing **four** policies:

- `replacement`: original random starts.
- `disjoint`: fixed-origin nonoverlapping T+1 tiles.
- `shifted`: seeded phase-shifted nonoverlapping tiles.
- `balanced`: roughly equal start counts in every eligible-start
  region, with replacement inside each region.

```bash
python examples/112_heldout_seed_coverage.py \
  --corpus data/tiny_story.txt --sampling-policy extended \
  --sampling-regions 4 --seed-counts 1,2,4 \
  --position-bins 8 --csv out/four_policies.csv \
  --json out/four_policies.json
```

The report distinguishes the balanced sampler's start counts from
actual token-position coverage, overlap and unsampled regions.
Every policy still draws from the same held-out corpus, but the
distributions differ.

The previous `--sampling-policy all` still compares just the
original three policies, so existing scripts remain reproducible.

Balanced sampling requires at least one requested window per
region. A small budget or too many regions fails with a clear
error instead of silently ignoring regions.

Run `python -m pytest tests/test_balanced_heldout_coverage.py -q`.

**Experiment:** Increase region count while holding the validation
batch budget fixed. At what count does sampling become infeasible?

**Checkpoint:** Why should start-region imbalance and token coverage
be measured separately even when every region receives starts?
