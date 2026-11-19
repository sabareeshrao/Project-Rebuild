# Lesson 203 — Where in the corpus did our validation windows land?

The previous experiments counted global unique positions touched by
sampled validation windows. But the *same total coverage* can be
concentrated at the beginning of the held-out corpus or spread
evenly along its length.

`analyze_position_coverage` divides corpus token positions into
equally spaced bins and counts, per bin:

- Distinct corpus-token positions touched by any sampled window.
- Total visits to token positions, including repeated windows.
- Repeated visits beyond the first; fraction of that bin covered.

Each window spans `block_size+1` token positions to include both
context and the next-token target. The per-bin distinct counts sum
to the exact globally deduplicated corpus coverage.

```python
profile = analyze_position_coverage(plans, position_bins=8)
print([segment.coverage_fraction for segment in profile.bins])
```

The result is a positional *coverage* histogram, not a histogram of
token values, model performance, or independent observations.
Counts do not change RNG state or re-sample validation windows.

Run `python -m pytest tests/test_position_coverage.py -q`.

**Experiment:** Give two plans the same global coverage but put
their touched windows in different halves of the corpus.

**Checkpoint:** Why might a high global coverage fraction still
hide an important unsampled region of validation text?
