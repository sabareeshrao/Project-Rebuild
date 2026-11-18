# Lesson 14: connect predictions to a differentiable loss

An untrained bigram model assigns mostly arbitrary next-token scores.
We need a way to tell it which predictions were wrong.

```text
input [B,T] -> embedding lookup -> logits [B,T,V]
                                         |
targets [B,T] --------------------> cross-entropy -> scalar loss
```

Call `loss.backward()` to ask PyTorch's **autograd** to calculate how
the loss changes with each embedding-table weight. Nothing has learned
yet; we only computed gradients. In the next lesson an optimizer will
**update** the parameters.

Run: `python examples/14_bigram_loss.py`

**Experiment:** Call `backward()` and inspect which rows of the
embedding table have nonzero gradients.

**Checkpoint:** What's the difference between computing gradients and
updating parameters with an optimizer?
