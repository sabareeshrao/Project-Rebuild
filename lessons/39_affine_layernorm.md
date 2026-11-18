# Lesson 39: LayerNorm learns how much to normalize

Normalizing to zero mean and unit variance is only the first step.
GPT-style LayerNorm also learns **one scale (gamma) and one shift
(beta) for each hidden feature**:

```text
output = gamma * normalize_features(x) + beta
gamma [C] initially ones; beta [C] initially zeros
```

These parameters are shared across all batch rows and token
positions, but each feature has its own pair. We explicitly compare
our educational implementation against PyTorch's `layer_norm` and
test that gradients reach both parameters.

Run: `python examples/39_affine_layernorm.py`.

**Experiment:** Set all scale values to zero. What happens to
normalized information? What does the bias still contribute?

**Checkpoint:** Why can LayerNorm avoid mixing information between
different time steps?

Later we can substitute a fused `nn.LayerNorm` for efficiency without
changing this layer's mathematical behavior.
