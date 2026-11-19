# Lesson 145 — Pin once, transfer one chunk, backpropagate

Our streamed optimizer helper now accepts two optional controls:

- `pin_memory=True` stages the logical batch in page-locked CPU memory.
- `non_blocking=True` requests asynchronous CUDA copies for its pinned
  microbatch slices.

Both are **disabled by default**. Nonblocking copies require pinning;
pinning requires a CUDA compute device. Incorrect combinations raise
clear errors instead of silently switching strategies.

```text
CPU logical batch → pin once (optional)
                       ↓
            current pinned CPU slice
                       ↓
             .to(cuda, non_blocking=...)
                       ↓
                   forward/backward
                       ↓
               next CPU slice …
                       ↓
               one optimizer update
```

Staging once is useful because repeatedly pinning each small slice
can introduce extra CPU memory copies. Copies and model operations
use the same CUDA stream, so this implementation does **not**
overlap transfers with computation on a separate CUDA stream.

Run `python -m pytest tests/test_streamed_transfer_controls.py -q`.

**Experiment:** On CUDA, compare optimizer outputs with pinning off
versus pinned + nonblocking mode, using identical initial weights.

**Checkpoint:** Why isn't `non_blocking=True` a guaranteed
end-to-end speed improvement for very small microbatches?
