# Lesson 233: how do GPT weights learn?

The forward pass computes logits and a loss. Backpropagation computes
one gradient for every learnable parameter that contributed to the loss.
We capture these gradients under **original nanoGPT tensor names**.

```text
tokens -> GPT -> loss -> backward() -> parameter gradients
                                   -> named tensor snapshots
```

Unlike a numerical value printed from the loss, a gradient describes
how changing a particular parameter would change that loss locally.

The token embedding and output projection **share one parameter** in
our upstream-compatible configuration. A gradient snapshot must count
that tied parameter once. We clone the gradients so later optimizer
steps cannot silently change our observations.

First create the capstone checkpoint using
`python examples/120_capstone_cli.py`, then run
`python examples/126_gradient_snapshot.py`.

**Experiment:** Change the input tokens. Which gradient norms change?

**Checkpoint:** Why does calling `zero_grad` *after* `backward`
erase the gradients needed for the next optimizer step?
