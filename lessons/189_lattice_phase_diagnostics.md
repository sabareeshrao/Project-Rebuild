# Lesson 189 — Report which nonoverlapping lattice was sampled

Shifting the disjoint-window grid is only useful as an experiment
when we can see the chosen phase. The `describe_disjoint_lattice`
helper inspects the plan's retained index positions and reports:

- The start phase modulo T+1 and number of feasible phases.
- The available full tiles on that phase and how many were selected.
- Prefix/suffix tokens left outside the complete tiled region.
- Fraction of corpus token positions touched by the selected windows.

The phase depends on the seeded plan and sample budget. Near full
capacity, fewer offsets can hold enough complete tiles. The
diagnostic rejects overlapping windows or mixed phases rather than
claiming they belong to one lattice.

```python
plan = prepare_shifted_evaluation_index_plan(
    tokens, batch_size=4, block_size=8, eval_batches=2, seed=19,
)
detail = describe_disjoint_lattice(plan)
print(detail.phase, detail.candidate_tiles_at_phase)
```

A phase/coverage report is **not** proof of unbiased sampling or
independent observations. Shifted plans may reuse positions across
different seeds, and larger sample budgets can change the set of
feasible phases.

Run `python -m pytest tests/test_shifted_evaluation.py \
tests/test_lattice_diagnostics.py -q`.

**Experiment:** Compare a near-capacity sample with a small sample
from the same corpus. What happens to feasible phase count?

**Checkpoint:** Why should we record the *actual* selected phase
rather than only the random seed?
