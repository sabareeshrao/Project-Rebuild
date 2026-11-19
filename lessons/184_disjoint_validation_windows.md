# Lesson 184 — Sample nonoverlapping windows deliberately

Our original sampled evaluator draws random starts **with
replacement**. Unique starts can still produce overlapping windows.

For a controlled alternative, split the held-out token stream into
fixed contiguous tiles of length `T+1` (T input tokens plus the
next-token target). Shuffle those tile IDs once, without replacement,
and take the requested number of tiles.

```text
held-out tokens: [tile 0][tile 1][tile 2][tile 3] ...
                ← T+1 → ← T+1 →
seeded randperm → selected tiles, each at most once
```

`prepare_disjoint_evaluation_index_plan` creates the same
`EvaluationIndexPlan` type as ordinary sampled evaluation. Its
starts are deterministic for a given seed; it does not advance the
global RNG, and every selected interval is disjoint **within that
seed's plan**.

Crucial caveats: the fixed lattice excludes nonaligned start
positions and any short tail, so this changes the sampling
distribution. Disjoint windows do not make one held-out corpus
independent observations, and separate seeds can reuse tiles.
Oversized sample requests raise an error instead of silently
repeating positions.

Run `python -m pytest tests/test_disjoint_evaluation.py -q`.

**Experiment:** Try a corpus with four full tiles and request five
windows. Explain why a hard failure is better than repeating a tile.

**Checkpoint:** What potential bias does always anchoring the
nonoverlapping tiling at position zero introduce?
