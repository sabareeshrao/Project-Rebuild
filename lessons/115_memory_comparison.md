# Lesson 115 — Compare estimated tensors with measured CUDA peaks

We now connect two distinct ideas:

1. **Tensor budget**: unique weights + buffers + estimated gradients
   and AdamW moments. This estimate works on CPU and CUDA.
2. **Allocator peak**: measured *additional* CUDA allocations during
   one optimizer step, over a baseline that already includes model
   parameters and AdamW moments. This is available only on CUDA.

```text
                 manual fused GPT       SDPA GPT
                     |                     |
          same architecture, same shapes, fresh training run
                     |                     |
            estimated tensor storage (all devices)
                     |                     |
           measured step allocation peaks (CUDA only)
```

Both implementations have the same parameter structure, but different
attention kernels can allocate different temporary buffers.

On CUDA, we execute an untimed warmup optimizer step *before* resetting
the memory peaks, so first-time optimizer moment allocation doesn't
mislead the comparison. Runs are sequential and release resources.

Run `python examples/90_memory_comparison.py --device cpu`.
On a supported GPU try `--device cuda --precision bf16`.

**Important:** The CPU output does **not** claim to measure peak VRAM.
GPU results include only allocations tracked by PyTorch's CUDA memory
allocator, not complete device memory and not always third-party
kernel workspaces. Allocator caching, warmup, and process state affect
measurements.

**Checkpoint:** Why shouldn't you directly equate
`estimated_training_bytes` with CUDA `peak_memory_allocated`?
