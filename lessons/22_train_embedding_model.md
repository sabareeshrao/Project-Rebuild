# Lesson 22: train the position-aware model

We can reuse the optimizer from our bigram lessons because both models
provide `train()` and `loss(inputs, targets)`. A small Python
`Protocol` documents the shared interface, rather than tying our
optimizer helper to a particular model class.

To test whether our new position embeddings matter, use this deliberately
simple training pattern:

```text
inputs:   0  0  0  0
targets:  0  1  0  1
position: 0  1  2  3
```

The **token IDs are identical at every position**, so a context-free
bigram table cannot assign four different outputs. Our position-aware
model **can**, because absolute position embeddings add a different
vector at each position.

Run: `python examples/22_train_embedding_model.py`

**Experiment:** Remove the position-vector addition from the forward
pass and rerun training. Can the model fit alternating targets?
Restore the addition afterward.

**Checkpoint:** How is *position awareness* different from *context
awareness*? What further information will causal self-attention provide?

The example proves a concept on a tiny repeated pattern, not general
language understanding or an improvement over nanoGPT.
