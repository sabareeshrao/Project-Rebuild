# Lesson 205 — Require minimum coverage in *every* corpus region

Global held-out token coverage measures a union, but it can
hide complete gaps. Covering 60% of a corpus does not imply
that any particular region was seen.

A regional target requires that **each** of B positional bins
have at least a chosen fraction of its token positions touched.

```text
20 tokens, five equal bins, minimum 50% per bin
observed distinct positions: [4, 4, 4, 0, 0]
required positions:          [2, 2, 2, 2, 2]
shortfall:                   [0, 0, 0, 2, 2]
target reached? NO
```

`assess_position_target` accepts the per-seed positional growth
curve and identifies the first already-selected seed prefix
meeting the goal in **all** bins. If no prefix reaches it,
the result names the missing positions in each bin and does not
add new seeds or guess when the target might be reached.

This is an experiment-design diagnostic, not a calibrated
statistical stopping rule, independence guarantee, or measure
of language-model quality.

Run `python -m pytest tests/test_position_targets.py -q`.

**Experiment:** Compare a 60% *global* coverage goal with a 50%
*per-bin* goal on a sample concentrated in the first three bins.

**Checkpoint:** Why can the regional goal fail even when a
less demanding-looking global goal succeeds?
