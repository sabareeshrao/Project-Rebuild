# Lesson 105 — Fix an initialization regression before optimizing

While reviewing the model ahead of SDPA integration, we found a real
syntax error: a duplicated `if` statement left the bias-initialization
operation incorrectly indented. This made the GPT module impossible
to import. We removed the duplication.

We also fixed a separate bug for `bias=False`: LayerNorm has no bias
tensor in that configuration. Calling `nn.init.zeros_(None)` is invalid.

```python
if isinstance(module, nn.Linear) and module.bias is not None:
    nn.init.zeros_(module.bias)
# and likewise for optional FeatureLayerNorm bias
```

The intent remains unchanged: initialize existing biases to zero, but
never try to initialize parameters that have deliberately been omitted.

Run `python -m pytest tests/test_initialization.py -q`.

**Experiment:** Deliberately introduce the extra `if` in a scratch
file. Which Python exception appears before any model code can run?

**Checkpoint:** Why is it safer to explicitly check optional
parameters instead of assuming every layer has a bias tensor?
