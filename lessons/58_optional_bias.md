# Lesson 58: optional bias in normalization and the MLP

In original nanoGPT, a configuration flag can disable the learned
bias in LayerNorm and all linear layers. This affects both the forward
calculation and the names available in a checkpoint.

We first make our reusable LayerNorm and token-wise GELU network
support `bias=False`, while preserving the earlier default.
`initialize_gpt_weights` also handles missing linear biases.

Run: `python -m pytest tests/test_upstream_bias_modules.py`.

**Experiment:** Compare `state_dict().keys()` with bias enabled
and disabled. Which keys disappear?

**Checkpoint:** Why must an absent bias be represented by `None`
instead of a frozen zero-valued Parameter?
