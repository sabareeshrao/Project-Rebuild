# Lesson 40: process each token with a feed-forward MLP

Self-attention lets tokens **exchange information**. A transformer
also uses a **position-wise feed-forward neural network** to process
the resulting information at each token independently.

```text
hidden [B,T,C]
 -> Linear(C,4C) -> GELU -> Linear(4C,C) -> dropout
 -> processed features [B,T,C]
```

The hidden width first expands to four times C, then returns to C.
GELU provides nonlinearity: without it, two linear layers would
collapse into one affine transformation. This educational version
uses PyTorch's default exact GELU.

The MLP **does not mix time steps**. Its weights are reused at every
position. Attention is responsible for mixing information across
tokens, while the MLP transforms features within each token.

Run: `python examples/40_feedforward.py`.

**Experiment:** Change only the final input token and verify earlier
MLP output vectors remain the same.

**Checkpoint:** Why do we need nonlinear transformations in addition
to attention, and what does the dropout module do in evaluation mode?
