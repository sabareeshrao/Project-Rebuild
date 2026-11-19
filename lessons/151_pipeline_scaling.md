# Lesson 151 — How pipeline bottlenecks change with workload size

The slowest stage is not necessarily constant. A tiny GPT may spend
proportionally more time in Python and CPU batch preparation, while
a deeper model can shift its bottleneck toward backward and AdamW.

We now sweep batch size and depth with controlled seed, context and
model width. Each row measures the same three stages *in one process*:

```text
sample token windows → transfer batch → GPT forward/backward/AdamW
```

```bash
python examples/102_pipeline_scaling_sweep.py \
  --batch-sizes 1,2,4 --depths 1,2,4 --context 8 \
  --csv out/pipeline_sweep.csv --json out/pipeline_sweep.json
```

The script limits the grid to 12 shapes to prevent accidental long
runs. CSV and JSON include all stage medians, tokens/update, parameter
count, seed and Python/PyTorch/device metadata. CPU mode is supported;
CUDA mode uses synchronized stage boundaries to ensure attribution.

**No speedup or trend is asserted in tests.** Even after fixing seeds,
wall-clock latency depends on thread scheduling, GPU clock rates,
allocator state and concurrent work. A larger workload need not have
a proportionally larger end-to-end time.

Run `python -m pytest tests/test_pipeline_scaling_sweep.py -q`.

**Experiment:** Hold batch size at 2 and increase model depth.
Does the update stage become a larger part of total latency?

**Checkpoint:** Why can a pipeline that is input-bound for one model
be compute-bound after only changing the number of transformer layers?
