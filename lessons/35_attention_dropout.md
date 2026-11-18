# Lesson 35: regularize multi-head outputs with dropout

**Dropout** randomly sets activations to zero **only during training**.
During evaluation, it passes activations through unchanged. It can
reduce reliance on particular features and help resist overfitting.

Our initial module applies dropout **after the multi-head output
projection**:

```text
parallel heads -> concatenate -> output projection -> dropout
```

A dropout rate of 1.0 is deliberately used in the example to make
training/evaluation behavior obvious: training returns zeros, while
evaluation returns normal projected features. In practice use a
smaller value, often 0.0–0.1 in tiny learning experiments.

Run: `python examples/35_attention_dropout.py`

**Experiment:** Change dropout to 0.5 and run twice in training
mode. Switch to `eval()` and compare outputs.

**Checkpoint:** Why is it important to use `model.eval()` when
measuring held-out loss?

**Important distinction:** nanoGPT also supports dropout *on
attention probabilities*. This commit introduces output dropout
only; probability dropout will be a separate concept later.
