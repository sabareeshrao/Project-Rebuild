# Lesson 78: prevent exploding updates with gradient clipping

A model's backpropagated gradient can occasionally have a very large
**global L2 norm**. A large update can destabilize training.

After `loss.backward()`, but **before** `optimizer.step()`, we can
scale every gradient by the same factor if their combined norm exceeds
a configured threshold:

```text
loss -> backward -> clip_grad_norm_(parameters, max_norm) -> AdamW update
```

Our `train_step(..., max_grad_norm=...)` helper uses PyTorch's
`clip_grad_norm_`. It rejects invalid thresholds and raises if the
gradients contain non-finite values when clipping is enabled.

We default to **no clipping** (`None`) to preserve earlier lessons.
Upstream nanoGPT also clips gradients after backpropagation, before
its optimizer update.

Run: `python examples/75_gradient_clipping.py`.

**Experiment:** Compare the single-parameter update with and without
a maximum norm of 0.1.

**Checkpoint:** Why do we clip gradients *before* the optimizer step
rather than clipping model weights afterward?
