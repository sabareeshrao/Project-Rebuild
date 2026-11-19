# Lesson 154 — Reuse window metadata without duplicating the corpus

The original batch sampler builds a tiny range of context offsets
for every draw. A reusable `CachedWindowSampler` instead holds the
`unfold` view from Lesson 153:

```python
sampler = CachedWindowSampler(tokens, block_size=8)
x, y = sampler.sample(batch_size=4, generator=rng)
```

Construction neither draws random numbers nor copies the whole corpus.
Each `sample` call still draws one tensor of random starts, then
gathers only the selected windows. We verify *multiple consecutive*
batches against `sample_batch` with separate identically seeded
generators, including identical generator state after each batch.

The table is a **view**, not a materialized giant list of contexts.
Its storage shares the original CPU or CUDA token tensor. Token IDs
and next-token targets remain unchanged, which is much more important
than winning a microbenchmark.

Run `python -m pytest tests/test_window_sampler.py -q`.

**Experiment:** Sample five batches with each implementation. Compare
the random generator's state after every draw.

**Checkpoint:** Why is matching just the first sampled batch
insufficient to prove that resume/restart behavior will be preserved?
