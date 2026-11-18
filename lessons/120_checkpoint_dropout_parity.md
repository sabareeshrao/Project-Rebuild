# Lesson 120 — Determinism when checkpointing recomputes dropout

Why does checkpointing need more than identical tensor shapes?

Dropout consumes random numbers during forward. During backward,
checkpointing may **replay** the same forward operations. If those
random draws change, the recomputed activations no longer correspond
to the original loss and the gradient can be wrong.

We explicitly choose PyTorch's checkpoint option
`preserve_rng_state=True`, so recomputed operations reuse the
appropriate random-number state.

Our full-GPT test uses two identical SDPA models with dropout:

1. Set the **same seed** before the ordinary forward+backward.
2. Reset that seed before the checkpointed forward+backward.
3. Compare loss, every parameter's gradient and CPU RNG state.
4. Confirm `.eval()` output is unchanged and state-dict keys match.

Run:

```bash
python -m pytest tests/test_checkpointed_gpt.py -q
```

**Experiment:** Remove `preserve_rng_state=True` in a scratch
branch and observe whether the dropout gradients still agree.

**Checkpoint:** Why might a naive checkpoint re-execution advance
the RNG and change the effective training algorithm?

For CUDA, determinism depends on device, kernels and PyTorch settings;
GPU-specific replay still requires testing on real CUDA hardware.
