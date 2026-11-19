# Lesson 201 — Set an optional corpus-coverage goal in the seed sweep

The seed-coverage experiment can now report when a fixed sequence
of validation seeds first reaches a chosen fraction of corpus
positions:

```bash
python examples/111_seed_coverage_sweep.py \
  --corpus-tokens 1024 --seed-counts 1,2,4,8 \
  --coverage-target 0.8 --sampling-policy all \
  --csv out/coverage_goals.csv --json out/coverage_goals.json
```

The tool generates exactly the requested seed sequences, then
**retrospectively** identifies the first prefix that touches
enough distinct corpus positions. It does not adapt its seed
choice or carry out extra sampling when the target is missed.

A CSV row records whether the target was reached, the number of
seeds needed and any remaining token-position shortfall.
JSON additionally reports the first seed ID crossing the
threshold. If the target is not reached, these crossing fields
are null rather than made-up values.

The default CLI output remains compatible: target columns are
added only if `--coverage-target` is specified.

Coverage saturation is an experiment-design signal, not a
statistical stopping rule or generalization guarantee. All
samples still come from one corpus and can be dependent.

Run `python -m pytest tests/test_seed_coverage_targets_cli.py -q`.

**Experiment:** Compare goals of 20%, 50% and 100% with the
same seeds and sampling policy.

**Checkpoint:** What does it mean when the 100% goal is not
reached even though all planned validation seeds were evaluated?
