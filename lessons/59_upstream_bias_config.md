# Lesson 59: propagate the upstream bias switch through GPT

Upstream nanoGPT's `GPTConfig` includes a `bias` flag. It affects
LayerNorm, QKV, attention output projection, and both MLP layers.
We now propagate it through **all blocks** and the final normalization.

Use `GPTConfig(..., attention_impl="fused", bias=False)` for the
no-bias variant. Our earlier educational separate-head path is
unchanged, and deliberately disallows this option because its
projection layout is not equivalent to upstream's fused path.

For a tied fused GPT with no biases, our parameter count becomes:

```text
V*C + T*C + C + L*(12*C*C + 2*C)
```

Run: `python -m pytest tests/test_upstream_bias_config.py`.

**Experiment:** Compare exported state-dict keys with `bias=True`
and `bias=False`. Which bias tensors disappear?

**Checkpoint:** Why must the model and checkpoint agree on this flag
before strict parameter loading?
