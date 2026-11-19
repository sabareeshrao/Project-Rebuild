# Lesson 198 — How many new tokens did each validation seed cover?

A cumulative coverage plot hides an important detail: one seed may
touch many tokens but introduce very few *new* token positions.

The new `analyze_coverage_growth` returns a step for each sampled
validation seed, in the supplied order:

- **New** positions absent from all earlier seed plans.
- Positions unique within this plan but **already covered** by a
  previous seed.
- Repeated token positions **within** the new seed's own windows.
- Cumulative unique coverage and corpus coverage fraction.

For each seed, these three position counts add to the number of
window-token positions sampled by that seed.

```text
Seed 10: [0..3], [8..11]  → 8 new positions
Seed 11: [0..3], [4..7]  → 4 new, 4 previously covered
Seed 12: [0..3], [0..3]  → 0 new, 4 previously covered, 4 within-seed repeats
```

The analyzer uses the existing cross-seed interval accounting
for each prefix; it never resamples starts or advances RNG.
A marginal gain of zero is a coverage observation, not proof
that the seed's **loss evaluation** contains no useful information.

Run `python -m pytest tests/test_coverage_growth.py -q`.

**Experiment:** Reverse the order of seeds 10 and 11. Which values
change, and which global final values remain the same?

**Checkpoint:** Why are marginal gains order-dependent even when
the final global corpus union is not?
