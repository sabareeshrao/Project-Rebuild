# Lesson 98 — FP32, BF16, FP16 and autocast

The model's *parameter storage* can remain FP32 while many computational
operations temporarily use narrower floating point formats.

- **FP32**: the existing behavior, no autocast.
- **BF16**: 8-bit exponent, smaller mantissa; generally no gradient scaler.
  We allow CPU and supported CUDA hardware.
- **FP16**: narrower exponent; small gradients can underflow, so CUDA
  training will later need a gradient scaler. We reject CPU FP16.

`resolve_precision(device, mode)` checks support before training begins.
`autocast_context(device, mode)` enables autocasting only around forward
calculations and loss, not around optimizer updates.

```python
with autocast_context(torch.device("cpu"), "bf16"):
    z = torch.mm(a, b)
```

Run `python -m pytest tests/test_precision.py -q`.

**Experiment:** Compare `torch.mm` output dtype inside vs. outside
an autocast context.

**Checkpoint:** Why might BF16 be more numerically stable than FP16 even
though both occupy 16 bits?
