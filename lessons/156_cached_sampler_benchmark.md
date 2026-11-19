# Lesson 156 — Is caching token-window metadata actually faster?

Caching is a performance *hypothesis*. We must measure it, not assume
it helps. The new CPU experiment runs the original indexed sampler
and our cached `unfold` sampler on the same token stream and RNG seed.

```bash
python examples/104_cached_sampler_benchmark.py \
  --batch-sizes 4,16,64 --contexts 4,16,64 \
  --iterations 10 --warmup 3 \
  --csv out/cached_sampler.csv --json out/cached_sampler.json
```

For each configuration it first proves that several consecutively
sampled batches and their final RNG states match. Only then does it
time each implementation with identical iteration counts and warmup.

The CSV reports median milliseconds, ratio, and the underlying corpus
and unfolded-view storage sizes. The view is not a duplicated corpus.
The report explicitly says **no model training or file loading**
was included.

Differences in Python overhead, CPU cache behavior and tensor shapes
may favor either sampler. Never assert a fixed speedup in correctness
tests.

Run `python -m pytest tests/test_cached_sampling_benchmark.py -q`.

**Experiment:** Increase T while keeping B constant, then increase B
while keeping T constant. Does the best sampler change?

**Checkpoint:** Why must a benchmark verify output parity *before*
reporting a latency ratio?
