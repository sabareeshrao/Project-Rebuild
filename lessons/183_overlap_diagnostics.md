# Lesson 183 — Measure duplicate and overlapping validation windows

Random sampling **with replacement** can choose the same context
twice, and even distinct context starts can share token positions.

For context length T, each input/target window touches **T+1**
consecutive corpus tokens. We now calculate duplicate starts,
neighboring interval overlaps, total touched token positions,
unique covered positions and repeated positions.

```python
from rebuild_gpt.window_overlap import analyze_window_starts
report = analyze_window_starts(
    [0, 0, 3, 10], block_size=3, corpus_token_count=20,
)
assert report.repeated_window_token_positions == 5
```

The `overlapping_neighbor_pairs` field counts overlapping
*neighboring* windows after sorting, not all possible pairs.
`coverage_fraction` describes observed corpus-token coverage, not
statistical independence or generalization.

We also fix the saved-checkpoint CLI JSON direction summary so it
explicitly includes `total` and `improved_fraction` alongside
improved/worsened/tied counts; dataclass properties do not appear
automatically in `asdict`.

Run `python -m pytest tests/test_window_overlap.py \
tests/test_saved_checkpoint_comparison_cli.py -q`.

**Experiment:** Compare [0,4,8] with [0,0,8] at T=3.

**Checkpoint:** Why can unique start positions still correspond to
substantial overlap in the tokens being evaluated?
