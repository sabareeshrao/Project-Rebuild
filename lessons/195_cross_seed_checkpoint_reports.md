# Lesson 195 — Export cross-seed coverage in saved GPT comparisons

The saved-checkpoint CLI already reports the overlap of windows
within each validation seed. It now also exports
`cross_seed_overlap` in JSON.

```bash
python examples/109_compare_saved_checkpoints.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt --sampling-policy disjoint \
  --seed-count 4 --json out/paired_coverage.json
```

The report distinguishes `repeated_positions_within_seeds`
from `repeated_positions_between_seeds`, plus the globally
unique touched positions and corpus coverage fraction.

Dataclass-computed fields such as coverage fraction and total
repeated positions are deliberately serialized explicitly.
The console also displays global distinct coverage and
cross-seed reuse.

These values can reveal an evaluation that touches the same
small part of held-out text many times despite different seeds.
They do not imply a calibrated effective sample size or a
statistically valid significance test.

Run `python -m pytest tests/test_cross_seed_checkpoint_cli.py -q`.

**Experiment:** Compare one seed to four seeds at the same
per-seed budget, keeping the corpus fixed.

**Checkpoint:** Why can a disjoint per-seed policy still produce
repeated token coverage when multiple seeds are evaluated?
