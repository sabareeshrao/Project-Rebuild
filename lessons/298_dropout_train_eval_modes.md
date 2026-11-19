# Lesson 298 — Dropout depends on whether GPT is training

Dropout randomly zeros selected activations *during training* to
regularize a model. At inference time, the same model should return
identical logits for identical tokens and weights.

This lesson checks both behaviors on a tiny GPT with nonzero dropout:

```text
eval:  tokens -> same logits on repeated calls
train: tokens -> different randomly masked activations -> varying logits
```

The check temporarily changes the model's mode and samples under a
controlled seed. It then restores both the original mode and the
global PyTorch RNG state, even on failure.

Run `python examples/191_dropout_modes.py`.

**Experiment:** Disable every Dropout module while leaving the
configuration's probability nonzero. Why should the negative test
reject this model?

These checks confirm local train/eval behavior, not training quality.
