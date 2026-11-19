# Lesson 143 — Why pinned CPU memory matters for CUDA copies

Normal CPU tensor storage uses pageable host memory. CUDA can transfer
data from **page-locked (pinned)** host memory with more predictable
asynchronous transfer semantics.

`stage_cpu_batch(x, y, device=..., pin_memory=True)` deliberately
pins the *whole logical CPU batch only once* before splitting it
into microbatches. The pinned tensors **stay on CPU** until explicitly
transferred to CUDA. Pinning itself may copy the data, so it has an
overhead and does not automatically make a training run faster.

```text
CPU sample [B,T] → pin host pages once → CPU microbatch views
                                           ↓
                                  transfer current chunk
```

Pinning is optional and requires an available CUDA accelerator.
The default mode does not initialize or query CUDA and returns the
original CPU tensors without copying.

Run `python -m pytest tests/test_pinned_batch.py -q`.

**Experiment:** On a CUDA machine, check `tensor.is_pinned()`
before and after staging. Verify values are unchanged.

**Checkpoint:** Why should a benchmark include the cost of *pinning*
and not measure just the subsequent GPU copy?
