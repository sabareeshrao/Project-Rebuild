# Lesson 104 — Prove optimized attention still learns the same task

An optimization is only useful if it preserves the intended mathematics.
With the **same weights** and dropout disabled, we compare:

1. Forward outputs for several context lengths.
2. Input gradients and parameter gradients after backward.
3. Causality when future token representations are modified.
4. Deterministic evaluation (SDPA must use `dropout_p=0` in eval).
5. Active dropout during training.

Gradient equality is approximate: SDPA and explicit matrix operations
can follow different floating-point execution orders. We use appropriate
tolerances rather than claiming bit-for-bit identity.

```text
same Q,K,V,weights → manual attention → outputs/gradients
                  → SDPA attention   → outputs/gradients
                                    → numerical comparison
```

Run `python -m pytest tests/test_sdpa_attention.py -q`.

**Experiment:** Increase the sequence length or try BF16 on compatible
hardware; compare numerical error and consider why precision matters.

**Checkpoint:** Why does matching the forward pass alone fail to prove
that an optimized attention kernel is suitable for training?
