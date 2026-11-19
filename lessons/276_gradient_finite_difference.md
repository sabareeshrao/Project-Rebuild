# Lesson 276: validate GPT backprop without trusting backprop

The earlier lessons compared gradients to original nanoGPT. Today we
use an **independent numerical approximation**: perturb a single model
weight and measure how the loss changes.

For one scalar parameter `w` and a small `epsilon`:

```text
numerical gradient = (loss(w + epsilon) - loss(w - epsilon)) / (2 epsilon)
autograd gradient  = d(loss)/dw
```

Those should agree when the analytical gradient is correct. Using a
float64 model makes subtraction of nearly equal losses reliable.
Eval mode turns off dropout. The original weight and model mode are
restored even if the check fails.

Run: `python examples/169_gradient_check.py`

The test suite also installs an intentionally corrupted gradient hook
that doubles the backward gradient; the independent check catches it.

**Question:** Why do tiny numerical differences become unreliable if
`epsilon` is too small, especially in float32?
