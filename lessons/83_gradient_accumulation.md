# Lesson 83: simulate a larger batch with gradient accumulation

A GPU may not have enough memory to process a large batch at once.
**Gradient accumulation** splits it into smaller microbatches, adds
their gradients, and updates the model **once**.

For K equally sized microbatches, compute:

```text
optimizer.zero_grad()
for microbatch in K:
    (loss(microbatch) / K).backward()
clip_grad_norm_(...)  # optional; once, after all microbatches
optimizer.step()      # once
```

The helper `train_accumulated_step` requires matching nonempty
microbatch shapes and returns their average loss. Dividing *before*
backpropagation prevents the gradient magnitude from growing by K.
Clipping the combined gradient once matches nanoGPT's training order.

Run: `python examples/80_gradient_accumulation.py`.

**Experiment:** Compare one four-item batch against two two-item
microbatches using identical weights and an SGD optimizer.

**Checkpoint:** Why must `optimizer.zero_grad()` and
`optimizer.step()` occur outside the microbatch loop?

This initial helper assumes equally sized microbatches; variable-size
batches would need sample-weighted losses rather than a simple mean.
