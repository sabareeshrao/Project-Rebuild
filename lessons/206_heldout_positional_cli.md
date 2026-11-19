# Lesson 206 — Profile validation blind spots on real held-out text

The existing held-out UTF-8 coverage experiment can now divide
the validation portion into positional bins. This is especially
useful for identifying whole regions that the sampled windows
did not touch.

```bash
python examples/112_heldout_seed_coverage.py \
  --corpus data/tiny_story.txt --seed-counts 1,2,4 \
  --position-bins 8 --minimum-bin-coverage 0.5 \
  --csv out/regions.csv --json out/regions.json
```

The optional positional output includes each region's boundaries,
unique covered token positions, repeated visits and covered
fraction. JSON additionally records per-region marginal gains
as seeds are added.

A regional target requires **every** bin to meet the specified
minimum fraction. The report states if or when the preselected
seed prefix met that target; unmet targets remain unmet, without
automatic resampling.

These flags are opt-in. Without `--position-bins`, older
summary columns and sampling behavior remain unchanged.

A covered token is not an independent observation, and a
well-covered corpus is not a guarantee of model generalization.

Run `python -m pytest tests/test_heldout_position_cli.py -q`.

**Experiment:** Compare two and sixteen positional bins for the
same seeded validation windows.

**Checkpoint:** Why does increasing the number of bins sometimes
reveal a previously hidden coverage gap?
