# Lesson 190 — Compare two GPTs on phase-shifted disjoint windows

The paired validation API now supports a third explicit policy:

```python
result = compare_paired_batches(
    baseline, candidate, val_tokens,
    seeds=(1,2,3), batch_size=4, block_size=8,
    eval_batches=2, sampling_policy="shifted",
)
```

With `replacement` (default), each random start is sampled
independently and windows may overlap. With `disjoint`, all tile
starts are multiples of T+1. With `shifted`, a seeded feasible
phase determines the tile grid before unique tiles are selected.

Each model sees **the same actual token windows** within each seed;
therefore candidate-minus-baseline differences remain paired.
The result also contains per-seed overlap counts and, for the two
disjoint modes, `lattice_by_seed` reports the actual phase and
candidate-tile capacity. Replacement stores `None` there.

The saved-checkpoint comparison inherits the same opt-in policy,
and original callers still default to replacement sampling.

A shifted lattice is not equivalent to drawing every possible
window start uniformly. It may reduce a fixed-origin artifact,
but phase feasibility depends on sample count and different seeds
can still share windows.

Run `python -m pytest tests/test_shifted_paired_validation.py -q`.

**Experiment:** Compare the same model with itself under all three
policies. Paired deltas must be zero regardless of the selected grid.

**Checkpoint:** Why is the sampled phase a property of the
evaluation plan that should be recorded alongside the random seed?
