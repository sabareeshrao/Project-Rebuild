# Lesson 167 — Decide whether caching is worth its setup and RAM cost

An evaluation cache can avoid repeated CPU token sampling, but building
the cache takes time and retains additional CPU memory. A fair
comparison includes all three quantities.

```bash
python examples/105_cached_eval_benchmark.py \
  --eval-batches 1,2,4 --eval-cache-max-mib 16 \
  --iterations 5 --warmup 2 \
  --csv out/cache_economics.csv --json out/cache_economics.json
```

Suppose a cache takes 12 ms to prepare. Regenerated evaluation takes
8 ms while cached evaluation takes 6 ms. The per-evaluation savings
are 2 ms, so the initial cost is only recovered after **six
evaluations**.

```text
break-even events = ceil(preparation ms / per-evaluation savings)
                  = ceil(12 / (8 - 6)) = 6
```

If cached evaluation is no faster per event, no finite break-even
count exists. The benchmark reports that as unavailable, rather
than promising a speedup.

The script now enforces an optional CPU cache budget **before**
running any configuration, and includes estimated net savings for
ten evaluation events alongside setup time, measured medians and
logical cache bytes.

These estimates use separate median measurements: they are
hypothetical arithmetic, not precise predictions. Timing variance,
GPU transfers, and changes in workload may shift the result.

Run `python -m pytest tests/test_cache_economics.py \
tests/test_cached_eval_benchmark.py -q`.

**Experiment:** Increase the number of validation batches and see
whether the break-even estimate improves or becomes unavailable.

**Checkpoint:** Why is comparing only cached and regenerated
per-evaluation latency insufficient for deciding whether to cache?
