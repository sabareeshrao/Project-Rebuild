# Lesson 84: weight decay applies to matrix weights, not all parameters

Original nanoGPT configures **AdamW** with two parameter groups:

- Trainable tensors with **2 or more dimensions** receive weight decay.
  These include attention/MLP matrices and token/position embeddings.
- Trainable tensors with **fewer than 2 dimensions** get zero weight
  decay. These include LayerNorm scale and shift, and linear biases.

The implementation filters out frozen parameters and uses each
Parameter **once**, even when token embeddings and output head weights
are tied. The new `configure_adamw` function accepts the learning
rate, weight decay, and Adam betas; CPU execution uses ordinary AdamW,
while CUDA may use fused AdamW when supported.

Run: `python examples/81_adamw_groups.py`.

**Experiment:** Set `weight_decay=0.0` and inspect the two groups.
Does it change which tensors belong to each group?

**Checkpoint:** Why does decaying a LayerNorm scale parameter have a
different regularizing effect from decaying a weight matrix?

**Next:** We'll make these groups opt-in for text training to avoid
silently changing the optimizer settings used in earlier lessons.
