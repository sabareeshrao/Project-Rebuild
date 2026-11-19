# Lesson 136 — Compare microbatch sizes at a fixed logical batch

To measure the impact of microbatching, the **total number of tokens
per optimizer update must stay constant**. Otherwise two benchmark
rows describe different amounts of work.

This script uses one fixed logical batch (for example, B=5, T=3),
the same synthetic input/target tensors and identical initial model
weights for every microbatch size:

```text
B=5, M=5  → 1 forward/backward → one AdamW update
B=5, M=2  → 3 forward/backward → one AdamW update
B=5, M=1  → 5 forward/backward → one AdamW update
              all use 15 target tokens/update
```

Run:

```bash
python examples/97_microbatch_benchmark.py \
  --logical-batch-size 6 --microbatch-sizes 1,2,3,6 \
  --context 8 --iterations 5 --warmup 2 \
  --csv out/microbatch.csv --json out/microbatch.json
```

The benchmark times complete training updates (forward, backward,
AdamW). On CUDA it also reports *additional PyTorch allocator peak
bytes* above the initialized optimizer baseline. On CPU the CUDA
memory value is **unavailable**, not a fabricated measurement.

Small microbatches can reduce live activations, but they introduce
more kernel launches and forward/backward passes. No speed or memory
improvement is guaranteed on every shape and device.

Run `python -m pytest tests/test_microbatch_benchmark.py -q`.

**Experiment:** Compare M=1,2,6 on the same GPU while holding the
logical batch fixed.

**Checkpoint:** Why is it misleading to compare throughput across
microbatch sizes when the effective tokens per update also change?
