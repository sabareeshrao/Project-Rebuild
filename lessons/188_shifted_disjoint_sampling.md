# Lesson 188 — Shift the nonoverlapping-window lattice by seed

Fixed disjoint validation tiles always start at token zero. That
avoids within-plan overlap but consistently excludes many possible
start positions. A seeded **phase-shifted** lattice is an alternative:

```text
T = 3, width = 4
fixed phase 0:   [0..3] [4..7] [8..11] ...
shifted phase 2:   [2..5] [6..9] [10..13] ...
```

`prepare_shifted_evaluation_index_plan` draws a phase using a
local CPU generator, then shuffles distinct tiles *within that
phase*. Every selected window includes T+1 token positions, and
none overlap inside one plan.

The phase must leave room for the requested number of windows.
If a near-capacity request only fits phase zero, the helper uses it
rather than sampling an invalid phase or silently repeating tiles.
Increasing the sample budget can change the set of feasible
phases, so it may also change the sampled prefix.

The phase-shifted policy **still changes the distribution** compared
with uniformly sampling every possible start. It does not make
windows independent, and multiple seeds may share token positions.

Run `python -m pytest tests/test_shifted_evaluation.py -q`.

**Experiment:** Compare the possible starts at phase zero and phase
two with a context length of three.

**Checkpoint:** Why does a seeded random phase reduce the
fixed-origin artifact without proving an unbiased estimator?
