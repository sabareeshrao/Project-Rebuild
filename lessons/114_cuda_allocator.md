# Lesson 114 — Measure peak PyTorch CUDA memory

An analytical parameter budget cannot predict temporary activations,
softmax matrices, or intermediate backward buffers. PyTorch exposes
two allocator measurements:

- **Allocated**: CUDA bytes currently occupied by tensor allocations.
- **Reserved**: bytes held by PyTorch's caching allocator, including
  unused cached blocks.

To measure one operation, synchronize the CUDA stream, record the
baseline, reset peak statistics, execute the operation, synchronize
again, then read the peaks. Do **not** reset after the operation.

```text
GPU existing weights → baseline
                     → reset peaks
                     → forward/backward/optimizer
                     → synchronize → peak allocated/reserved
```

`measure_cuda_peak()` intentionally rejects CPU rather than inventing a
fake VRAM measurement. It only measures the PyTorch CUDA allocator, not
entire GPU/driver VRAM. Baseline allocations (model and optimizer)
are reported separately from the peak increase.

Run `python -m pytest tests/test_cuda_memory.py -q`.

**Experiment:** On a GPU, execute a training step with batch size 2,
then batch size 8; compare *additional peak allocated* bytes.

**Checkpoint:** Why can reserved bytes remain high even after most
tensor allocations have been freed?
