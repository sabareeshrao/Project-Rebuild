# Lesson 130 — Export a repeatable workload grid

We now turn the pure workload estimator into a small command-line
experiment generator. No model or GPU is needed.

```bash
python examples/95_scaling_grid.py \
  --batch-sizes 1,2 --contexts 4,8,16 --depths 1,2,4 \
  --accumulation 2 --csv out/scaling_grid.csv
```

Every CSV row records B, T, L, heads, dtype, accumulation, tokens per
optimizer update and the sizes of explicit attention-score tensors.

This is useful for **planning** a benchmark: it exposes the shape
choices before we commit time and GPU memory to actual training.
It does *not* predict wall-clock time, total activation storage or peak
VRAM. The same formula can estimate hypothetical FP16 tensor bytes
even on a CPU-only computer without performing FP16 computations.

Each row is deterministic and ordered. A maximum of 64 configurations
is enforced by default.

Run `python -m pytest tests/test_scaling_grid_cli.py -q`.

**Experiment:** Generate a grid with and without `--accumulation 4`.
Which columns change while score storage remains the same?

**Checkpoint:** Why is a CSV of analytical estimates not a substitute
for a real CUDA allocator measurement?
