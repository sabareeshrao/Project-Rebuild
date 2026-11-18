# Lesson 118 — Why attention has quadratic score storage

For a batch of B sequences, H heads and T positions, manual causal
attention forms a score tensor of shape **[B, H, T, T]**.

A single float32 score matrix takes `B × H × T² × 4` bytes. Doubling
T therefore multiplies *this matrix's* size by four, even when the
model's parameter count stays unchanged. BF16 halves each element's
storage relative to FP32.

```text
Q [B,H,T,D] × Kᵀ [B,H,D,T]  →  scores [B,H,T,T]
                                    |
                              mask + softmax
```

This is an **analytical estimate for one score matrix**, not total
attention memory or GPU peak memory. The manual implementation may also
materialize probability tensors, gradients and temporaries; optimized
SDPA kernels may avoid the full quadratic intermediate.

Run:

```bash
python -m pytest tests/test_activation_memory.py -q
```

**Experiment:** Calculate bytes for B=4, H=8, T=128 and then T=256.

**Checkpoint:** Why can attention activation storage rise sharply
without changing the number of learned GPT parameters?
