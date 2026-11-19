# Lesson 193 — Separate within-seed and cross-seed validation overlap

A disjoint plan avoids overlapping token windows *within a seed*.
That does not prevent two different seeds from selecting some
of the same token positions.

The new `analyze_cross_seed_overlap` accepts multiple index
plans referencing the same CPU corpus and reports the global
union of covered token positions.

```text
Seed 1: [0..3]          [8..11]
Seed 2: [0..3]  [4..7]
Union:  [0..........11]
```

Both plans are nonoverlapping internally, yet they share four
token positions between seeds.

We distinguish repeated positions **within** each plan from
additional repetitions **between** plans:

```text
all repeated positions = total window-token positions - global union
                       = within-seed repeats + between-seed repeats
```

The counts are deterministic and require no new random draws.
Plan seeds must be unique and all plans must share the same
corpus tensor and context length.

This is descriptive corpus coverage, **not** an effective sample
size or confidence interval.

Run `python -m pytest tests/test_cross_seed_overlap.py -q`.

**Experiment:** Use two disjoint plans with identical starts and
compare their within-seed and between-seed repeats.

**Checkpoint:** Why is zero overlap within each seed insufficient
to claim that results from several seeds are independent?
