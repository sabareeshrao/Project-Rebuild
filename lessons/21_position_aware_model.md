# Lesson 21: compose the first GPT-like input path

We now join token embeddings, learned absolute position embeddings, and
a linear language-model output head.

```text
IDs [B,T]
   |-> tokens [B,T,C] --------+
   |-> positions [T,C] -------+-> sum [B,T,C] -> head -> logits [B,T,V]
targets [B,T] ---------------------------------------> cross-entropy
```

All three parameter tables receive gradients from the loss.

**Important limitation:** Adding position embeddings does **not**
allow one token to inspect any earlier tokens. At each position, logits
depend only on the current token and its absolute position. The
attention mechanism in future lessons will introduce context mixing.

Run: `python examples/21_position_aware_model.py`

**Experiment:** Change token 0 but leave tokens 1 and 2 unchanged.
Will the predictions at positions 1 and 2 change?

**Checkpoint:** Why is this model still not a transformer?
