# Lesson 138 — Yield one weighted microbatch at a time

A list of tensor views is cheap, but transferring *every* view to GPU
before starting backward can unnecessarily keep every input tensor
resident on the accelerator.

Our new `iter_weighted_microbatches` produces the same views **lazily**.
For a logical batch of five examples with microbatch size two, it emits:

```text
(x[0:2], y[0:2], weight=2/5)
(x[2:4], y[2:4], weight=2/5)
(x[4:5], y[4:5], weight=1/5)
```

These weights account for every target element, assuming the loss is
a mean over all target elements. This is essential for the smaller
last chunk.

The generator itself makes no tensor copies or device transfers.
A training helper can now move a single chunk to CUDA, compute and
backpropagate its loss, then let the chunk go out of scope before
processing the next one.

Run: `python -m pytest tests/test_weighted_microbatch_iterator.py -q`.

**Experiment:** Inspect `data_ptr()` for the first slice and original
tensor. Why do they share storage?

**Checkpoint:** Why is lazy iteration useful even when slicing a
batch already creates zero-copy PyTorch views?
