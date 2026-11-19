# Lesson 139 — Transfer a single microbatch to GPU at a time

Previously the trainer could split a logical batch into smaller chunks
but move all those chunks to the model device *before* any backward
work. That unnecessarily kept the whole logical input batch resident
on GPU.

The new `train_streamed_logical_batch_step` keeps logical inputs
on CPU and transfers **only the current chunk**:

```text
CPU logical B=7 → chunk 3 → GPU → forward/backward → release
                → chunk 3 → GPU → forward/backward → release
                → chunk 1 → GPU → forward/backward → release
                                              ↓
                                        one AdamW update
```

It uses the target-count weights from Lesson 138, supports CPU BF16
and CUDA FP16 gradient scaling, and clips gradients after accumulating
all chunks. The optimizer update still happens exactly once.

CPU tests compare SGD and AdamW parameters with the original
full-batch and buffered-microbatch paths. CUDA tests are skipped
unless a compatible GPU is available.

Run `python -m pytest tests/test_streamed_microbatch_step.py -q`.

**Experiment:** Use a model on CUDA with source tokens on CPU; inspect
device placement inside the model's loss function.

**Checkpoint:** Why can streaming reduce GPU-resident *input* memory
without guaranteeing a corresponding reduction in total GPU VRAM?
