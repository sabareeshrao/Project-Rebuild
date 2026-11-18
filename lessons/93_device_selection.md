# Lesson 93 — CPU versus CUDA: where do tensors live?

A PyTorch `torch.device` describes the machine that owns tensor storage.
GPU training only works when model parameters **and input tensors** are on
compatible devices. A tensor stored on CPU cannot be used directly as the
input to a model stored on CUDA.

We introduce a simple resolver:

```python
from rebuild_gpt.devices import resolve_device
resolve_device("cpu")   # always CPU
resolve_device("cuda")  # require CUDA; otherwise raise ValueError
resolve_device("auto")  # CUDA if available, CPU otherwise
```

This commit only **selects a device**. It does not yet move the model,
batches, or evaluations. Those are the next lessons.

Run: `python -m pytest tests/test_devices.py -q`

**Experiment:** Ask for CUDA on a computer without an NVIDIA CUDA driver.
Why should the program fail rather than silently train on CPU?

**Checkpoint:** What goes wrong if the input tensor and embedding weights
are on different devices?
