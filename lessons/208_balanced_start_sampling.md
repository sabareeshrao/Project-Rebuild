# Lesson 208 — Balance validation *starts* across corpus regions

Uniform random starts can leave some held-out corpus regions
unvisited in a small validation run. A new opt-in **balanced**
policy divides the valid start positions into equal-sized
regions and gives each approximately the same number of draws.

For W requested windows and R regions, each region receives
either `floor(W/R)` or `ceil(W/R)` starts. Windows are sampled
**with replacement within a region**, using a local seeded CPU
generator, so global training RNG is unchanged.

```python
plan = prepare_balanced_evaluation_index_plan(
    heldout_tokens, batch_size=4, block_size=8,
    eval_batches=4, seed=214, region_count=4,
)
```

It fails if there are fewer than R requested windows, or if the
corpus cannot support R nonempty valid-start regions.

**Caveat:** Equal counts of *start positions* do not imply equal
coverage of all corpus tokens. Context windows may cross region
boundaries or overlap one another. The policy changes the
sampling distribution relative to uniform random starts and
can bias validation estimates on some datasets.

Changing the evaluation batch count can change the allocation of
starts to regions, so sampled prefixes need not be preserved.

Run `python -m pytest tests/test_balanced_evaluation.py -q`.

**Experiment:** Compare one seed's start counts using balanced
sampling and with-replacement uniform starts.

**Checkpoint:** Why does balancing starts across the corpus
not guarantee an unbiased estimate of the original objective?
