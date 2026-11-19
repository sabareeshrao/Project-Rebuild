# Lesson 161 — Does caching validation batches save enough time?

Caching evaluation examples trades CPU sampling work for retained
host memory. But it also has an up-front preparation cost that a
fair performance report must not ignore.

We now benchmark three quantities:

1. Median cost to prepare both train and validation batch plans.
2. Median repeated validation latency when batches are regenerated.
3. Median repeated validation latency when precomputed batches are reused.

```bash
python examples/105_cached_eval_benchmark.py \
  --eval-batches 1,2,4 --iterations 5 --warmup 2 \
  --csv out/eval_compare.csv --json out/eval_compare.json
```

The experiment checks that losses match exactly before measuring,
and reports **retained CPU bytes** for both evaluation plans. It
does not run optimizer updates or include training, so any speedup
applies only to the sampled evaluation stage.

On GPU, transfer of the CPU batches still occurs at each evaluation
step. The cache does not store model outputs or accelerate backward.

Run `python -m pytest tests/test_cached_eval_benchmark.py -q`.

**Experiment:** Increase `--eval-batches` and compare cache size,
preparation time, and repeated evaluation time.

**Checkpoint:** Why can a cached evaluator lose time overall when
the run performs only one or two evaluation events?
