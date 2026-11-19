# Lesson 202 — Measure seed coverage on real held-out text

Our earlier seed-coverage experiment used a synthetic token stream.
We can now inspect the same sampling and overlap behavior on a
real UTF-8 text corpus's contiguous held-out portion, without
running GPT forward passes or optimizer updates.

```bash
python examples/112_heldout_seed_coverage.py \
  --corpus data/tiny_story.txt --batch-size 4 --context 8 \
  --seed-counts 1,2,4,8 --sampling-policy all \
  --coverage-target 0.5 \
  --csv out/heldout_coverage.csv --json out/heldout_coverage.json
```

The script uses the existing character tokenizer and contiguous
train/validation split, with the last 20% of characters held out.
Each policy samples exactly the requested seed streams. Reports
include held-out token count, global union of touched token
positions, cross-seed repetition, marginal new positions added
by the last seed, and optional target achievement.

JSON also preserves the marginal coverage trace of each seed,
the vocabulary size, context, seed selection and the explicitly
model-free nature of the experiment.

Every policy still samples from one held-out text. Coverage alone
cannot establish generalization, model quality, independent
validation observations, or a calibrated confidence interval.

Run `python -m pytest tests/test_heldout_seed_coverage.py -q`.

**Experiment:** Compare the synthetic-token coverage sweep with
real held-out text using the same corpus length and sampling
settings. Which coverage quantities should agree?

**Checkpoint:** Why can identical coverage patterns conceal very
different language-model validation losses?
