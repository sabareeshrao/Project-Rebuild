# Lesson 111 — Benchmark an entire GPT training update

Inference timing is **not** training timing. A training update includes:

```text
tokens → model forward → cross-entropy → backward gradients → AdamW step
```

We create two GPTs with *identical initial weights* and run the same
fixed input/target batches for the same number of warmup and timed
training updates.

The benchmark uses `time_operation`, which includes CUDA
synchronization around each completed update. It reports median
milliseconds, a latency ratio and tokens per second.

```bash
python examples/89_training_benchmark.py --iterations 5 --warmup 2
# Optional on supported hardware:
python examples/89_training_benchmark.py --device cuda --precision bf16
```

The models maintain independent AdamW state; updates during warmup
are included in the optimizer history but not the timing samples.
Dropout is zero and the input batch is fixed to reduce noise.

The benchmark **does not guarantee a speedup**, and optimizer
performance may dominate on small models. It is not intended as a
large-scale throughput claim.

Run `python -m pytest tests/test_training_benchmark.py -q`.

**Experiment:** Increase block_size from 8 to 32 and compare the
manual/SDPA ratio.

**Checkpoint:** Why can one attention backend be faster for inference
but have similar or slower time per *training step*?
