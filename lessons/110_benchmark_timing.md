# Lesson 110 — Benchmark the operation, not just the stopwatch

The first performance lesson measures wall-clock **latency** correctly.
`time_operation` calls an operation several times as warmup, then
reports median/min/max of independent timed samples.

```text
warmup calls → synchronize → sync → start → operation → sync → stop
                                  (repeat N times)
```

CUDA launches kernels asynchronously. Without synchronizing **after**
the operation, a CPU timer would record launch overhead rather than
completed GPU work. We also synchronize **before** each measurement to
avoid including previously queued GPU work.

Use the median rather than a single noisy observation. This function
deliberately doesn't use random seeds, clear memory caches, or reset
model weights: those are the benchmark author's responsibilities.

Run: `python -m pytest tests/test_benchmarks.py -q`.

**Experiment:** Increase the number of iterations in a CPU benchmark,
and inspect all samples rather than only the median.

**Checkpoint:** Why can a GPU operation appear extraordinarily fast
if you omit the final synchronization?
