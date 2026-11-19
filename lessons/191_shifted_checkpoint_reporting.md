# Lesson 191 — Include shifted-lattice provenance in saved comparisons

The saved-checkpoint comparison CLI now supports
`--sampling-policy shifted` alongside the original `replacement`
default and `disjoint` fixed-origin alternative.

```bash
python examples/109_compare_saved_checkpoints.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt --sampling-policy shifted \
  --eval-batches 3 --seed-count 2 \
  --csv out/shifted.csv --json out/shifted.json
```

CSV includes each seed's actual lattice phase, feasible phase count,
available tiles, repeated corpus-token positions and paired losses.
JSON includes fuller per-seed lattice and overlap metadata.

The key reproducibility detail is that feasible lattice phases
depend on corpus size, context and sample budget. Recording the
random seed alone does not make the selected phase obvious.

Ordinary with-replacement runs report a null lattice phase rather
than pretending to use a disjoint tiling. Identical checkpoints
must still produce exactly zero paired differences under every
sampling policy.

A shifted lattice is not a universal correction for sampling bias.
All rows remain descriptive observations on one corpus, not
independent statistical trials.

Run `python -m pytest tests/test_shifted_checkpoint_cli.py -q`.

**Experiment:** Repeat the exact same checkpoint comparison twice
with the same seeds; compare both the phases and losses.

**Checkpoint:** Why is a null lattice phase meaningful for the
original with-replacement sampler?
