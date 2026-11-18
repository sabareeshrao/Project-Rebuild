# Lesson 117 — Inspect operator-level work with torch.profiler

Timing a full training step gives one number, but not its causes.
`torch.profiler` can show what PyTorch spends time on:

- Embedding and linear-layer operations.
- Attention matrix operations or SDPA kernels.
- Backward/gradient operations.
- Optimizer updates and tensor allocation events.

Run:

```bash
python examples/92_torch_profiler.py --device cpu \
  --attention-impl sdpa --output out/profiles/gpt_trace.json
# Optional NVIDIA GPU:
python examples/92_torch_profiler.py --device cuda --precision bf16
```

The script writes a Chrome trace-compatible JSON file and prints a
CPU operator summary. Compare the **shape** and kinds of operations
between `--attention-impl fused` and `sdpa`.

```text
synthetic tokens → GPT step → torch.profiler → operator events + memory
                                                  |
                                         JSON timeline for analysis
```

**Critical distinction:** Instrumentation adds overhead. Do not compare
absolute profiled run duration with an unprofiled benchmark and conclude
the algorithm became slower. For performance comparison, use the
synchronized benchmarks from lessons 110–116.

Profiler traces may contain operation names, shapes, timing, stack or
other workload metadata. Inspect files before sharing them publicly.
The trace is created under `out/`, which is ignored by Git.

Run `python -m pytest tests/test_torch_profiler.py -q`.

**Experiment:** Generate one manual-attention trace and one SDPA trace.
Find the operator corresponding to causal attention in each.

**Checkpoint:** Why does a profiler help answer *where* time is spent,
while a benchmark answers *how much time* an operation takes overall?
