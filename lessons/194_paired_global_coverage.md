# Lesson 194 — Add corpus-wide coverage to paired GPT validation

Paired saved-model comparisons previously exposed per-seed overlap.
We now also retain the actual sampled index plans long enough to
calculate their **combined corpus coverage across all seeds**.

`compare_paired_batches` returns `cross_seed_overlap` with:

- Unique validation token positions touched by *any* seed.
- Duplicate starts and repeated positions *within* seeds.
- Additional repeated starts and token positions *between* seeds.
- The total window count and fraction of the held-out corpus covered.

The model objectives, loss order, sampling generator and default
policy remain unchanged. The overlap calculation does not run
another model forward pass or consume RNG.

A compelling edge case: if the held-out text contains exactly
four disjoint T+1 tiles and we sample all four with **two** seeds,
every seed has zero internal overlap. Yet the two seed plans share
every covered token position, and cross-seed repeated coverage is
a full corpus's worth.

Run `python -m pytest tests/test_paired_cross_seed_overlap.py -q`.

**Experiment:** Compare one seed to two seeds for exact-capacity
disjoint sampling. Does global unique coverage double?

**Checkpoint:** Why isn't sampling several disjoint plans on the
same corpus equivalent to collecting several independent corpora?
