# Lesson 19: turn hidden vectors into logits

We now have two different dimensions:

- **V**: vocabulary size (possible token IDs).
- **C**: embedding width (features used to represent each token).

A linear projection maps each vector of width C into V scores:

```text
IDs [B,T] -> embeddings [B,T,C] -> linear head -> logits [B,T,V]
```

With `bias=False`, the output projection contains `C * V`
learnable weights. Softmax and cross-entropy already exist from earlier
lessons. This is a **linear output head**, not attention.

Run: `python examples/19_output_projection.py`

**Experiment:** Compare `C=6` and `C=12` while keeping `V=4`.
How does the final logits shape change?

**Checkpoint:** Why do we need V prediction scores at the output, even
when the hidden representation only has C features?
