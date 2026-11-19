# Lesson 131 — Measure the cost of batch size and transformer depth

We now measure **complete training steps**, not just analytic
attention-score tensors or inference forward time.

The new benchmark varies B (batch size) and L (transformer depth)
while keeping T (context) and the hidden width fixed.

```bash
python examples/96_depth_batch_benchmark.py \
  --batch-sizes 1,2,4 --depths 1,2,4 --context 8 \
  --iterations 5 --warmup 2 --csv out/depth_batch.csv
```

Each variant initializes a fresh GPT, optimizer, and deterministic
synthetic microbatches. Its measured step includes forward,
backward, and an AdamW update. The script reports median milliseconds
and effective tokens per second.

```text
B × T × accumulation = tokens/update
median step duration → effective tokens/second
```

With `--accumulation 2`, two microbatches are evaluated sequentially
before one optimizer update. The timed section deliberately does not
include dataset loading or sampling, so throughput is **model training
compute throughput**, not full input-pipeline throughput.

PyTorch CUDA and CPU performance differ by hardware, backend, dtype,
shapes, and cache state. A faster result is never assumed.

Run `python -m pytest tests/test_depth_batch_benchmark.py -q`.

**Experiment:** Hold L fixed, double B, and observe whether tokens
per second scale linearly. Then hold B fixed and double L.

**Checkpoint:** Why isn't doubling the batch size guaranteed to
halve the time required to process a fixed number of tokens?
