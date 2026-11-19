# Lesson 132 — Record enough context to interpret benchmark numbers

A throughput number without its settings and environment is hard to
reproduce. Two machines might run the same Python script and obtain
different answers because of PyTorch kernels, CPU threads, precision,
model dimensions or system load.

The training benchmark now supports a seed and a structured JSON
report in addition to the existing CSV measurements:

```bash
python examples/96_depth_batch_benchmark.py \
  --batch-sizes 1,2 --depths 1,2 --context 8 \
  --seed 77 --iterations 5 --warmup 2 \
  --csv out/scaling_results.csv --json out/scaling_report.json
```

The JSON includes:

- Exact requested batch sizes, depths, context, accumulation, dtype,
  attention backend, iteration count and random seed.
- Python/PyTorch versions, resolved device, CPU thread count and
  a CUDA device name **only when CUDA is selected**.
- Every numeric measurement from the CSV.
- A reminder that the benchmark reuses synthetic batches and
  excludes dataset loading.

A seed improves repeatability but doesn't make performance timing
deterministic. CPU scheduling, GPU clocks and thermal conditions still
matter; on CUDA, kernels may also differ by hardware.

Run:

```bash
python -m pytest tests/test_benchmark_report.py \
  tests/test_depth_batch_benchmark.py -q
```

**Experiment:** Run the same benchmark twice and compare the result
JSONs. Which fields should match, and which might fluctuate?

**Checkpoint:** Why is a speedup claim less useful when the batch
size, precision or model depth is missing?
