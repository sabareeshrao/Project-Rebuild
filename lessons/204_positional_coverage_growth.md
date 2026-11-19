# Lesson 204 — Which regions benefit from the next seed?

A seed can add many distinct token positions while contributing
almost nothing to another part of the held-out corpus.
A global marginal-gain number hides this spatial imbalance.

`analyze_position_growth` measures newly touched positions
**within each positional bin**, using one common set of corpus
boundaries for all seed prefixes.

```text
20 held-out positions, five bins:
seed 1: [4, 0, 4, 0, 0] new positions
seed 2: [0, 4, 0, 0, 0] new positions
seed 3: [0, 0, 0, 0, 0] new positions
final:  [4, 4, 4, 0, 0] covered
```

Results also count how many *previously untouched* bins became
touched after each seed. These increments reconcile with the
global marginal coverage analyzer and with the final positional
coverage histogram.

Seed order changes who receives credit for a newly covered
position. It does **not** change the final union of positions.

Run `python -m pytest tests/test_position_growth.py -q`.

**Experiment:** Reverse two seed plans whose windows partly
overlap. Does the first seed receive more credit?

**Checkpoint:** Why are zero new positions in a bin and zero
independent information not the same concept?
