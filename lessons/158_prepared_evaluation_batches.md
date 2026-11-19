# Lesson 158 — Reuse the same sampled validation batches

Our existing sampled validation deliberately draws the **same batches
from a fixed seed** at every evaluation step. It redraws and gathers
their token windows each time, though, introducing avoidable CPU work.

We now provide `prepare_evaluation_batches` that builds this fixed
set once and stores input/target tensors on CPU:

```text
validation tokens + seed → [batch 1, batch 2, ..., batch K]
                                          |
                             reuse at each evaluation event
```

The data stay on CPU, even if the model later executes on CUDA.
The plan does not change the training generator or global torch RNG.
Unit tests compare every prepared batch with the legacy `sample_batch`
sequence and verify that the global RNG state is unchanged.

**Memory tradeoff:** each cached batch has both input and target
tensors, so retained storage is approximately
`2 × eval_batches × batch_size × block_size × 8` bytes for
64-bit token IDs. The size can grow considerably with large batches
and many validation samples.

Run: `python -m pytest tests/test_eval_cache.py -q`.

**Experiment:** Prepare four batches twice with the same seed.
Compare their token IDs, and calculate retained bytes.

**Checkpoint:** Why is it safe to reuse evaluation examples between
training steps but generally wrong to reuse the same training
examples *instead of sampling new ones* without intending to?
