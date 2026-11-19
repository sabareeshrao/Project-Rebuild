# Lesson 144 — Nonblocking copies must have the right prerequisites

A CPU-to-CUDA transfer can request `non_blocking=True`, but it does
not automatically mean compute and transfer will overlap.
It is most useful with **pinned CPU source memory** and correct stream
management.

Our explicit `transfer_microbatch` validates that:
- Both sources are tensors.
- The target is CPU or CUDA.
- A nonblocking transfer targets CUDA.
- Both sources are CPU tensors that are already pinned.

The normal, blocking transfer path works on CPU and CUDA and remains
the default. The helper uses PyTorch's current CUDA stream: a
subsequent model forward on that same stream is ordered after the
copy. It **does not** create a side stream or promise overlap.

```text
pinned CPU slice → .to(cuda, non_blocking=True) → GPT forward
                        same CUDA stream ordering
```

Run `python -m pytest tests/test_microbatch_transfer.py -q`.

**Experiment:** Compare the values of CPU token IDs before and after
a transfer to CUDA. Synchronize before reading the result on CPU.

**Checkpoint:** Why does `non_blocking=True` not imply that transfer
and the current model's computation actually happen in parallel?
