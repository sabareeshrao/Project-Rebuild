# Lesson 94 — Deterministic evaluation across devices

Validation must avoid changing the training random-number stream.
We sample windows from the source corpus **on CPU** using a local
`torch.Generator`, and then transfer each `(x, y)` batch to the
device that holds the model parameters.

```text
CPU text tokens → seeded CPU batch sampler → x,y → to(model.device)
                                                  → model evaluation
```

This makes GPU training possible without changing the training corpus
representation or the deterministic choice of input windows.

Both `model.eval()` and `torch.no_grad()` are essential. We restore the
model's earlier train/eval mode after calculating the loss.

Run: `python -m pytest tests/test_evaluation.py -q`

A GPU-specific test runs only when CUDA is available; it is skipped on CPU.

**Experiment:** Use two different validation seeds. Why can the measured
loss change even if the model parameters stay identical?

**Checkpoint:** Why is moving only the input tensor to CUDA insufficient
when the target tensor is consumed by the GPU loss function?
