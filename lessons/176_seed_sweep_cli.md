# Lesson 176 — Measure validation seed sensitivity on real text

The sample evaluation estimates held-out GPT loss from a selection of
windows. With different seeds, the selected windows change even if
the model stays fixed.

The new diagnostic script loads a UTF-8 corpus, creates a
character-level vocabulary, selects a contiguous held-out split and
evaluates an **untrained** tiny GPT across several independent RNG
seed streams:

```bash
python examples/107_validation_seed_sweep.py \
  --corpus data/tiny_story.txt --seed-count 5 \
  --batch-size 4 --context 8 --eval-batches 4 \
  --csv out/seed_sweep.csv --json out/seed_sweep.json
```

The CSV preserves individual seed means. JSON records the model seed,
sampling seeds, hardware/runtime metadata and the descriptive mean,
spread and standard error of those seed means.

This experiment is **not a training result** and not a calibrated
confidence interval. The different draws all sample windows from the
same fixed held-out text, often with overlap. Choosing more seeds
shows sampling sensitivity without creating independent held-out data.

Run `python -m pytest tests/test_validation_seed_sweep.py -q`.

**Experiment:** Compare one validation batch per seed with eight
validation batches per seed.

**Checkpoint:** Why do sampling seeds and model initialization seed
have to be recorded separately?
