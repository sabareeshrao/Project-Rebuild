# Lesson 109 — From attention parity to optimizer parity

Testing a single attention layer is useful, but a complete GPT contains
multiple blocks, residual connections, LayerNorm and a tied embedding
matrix. A tiny numerical difference at one layer can propagate through
the network.

This lesson compares two models with identical parameters:
- packed QKV followed by explicit masked attention;
- identical packed QKV followed by PyTorch SDPA.

We verify cross-entropy losses, gradients for *every named parameter*,
and parameters after one AdamW optimizer update, for multiple contexts.

```text
same initial weights + same (x, y)
         |                      |
    manual GPT              SDPA GPT
         |                      |
       loss                   loss
       grad                   grad
      AdamW                  AdamW
         \                     /
           compare tolerances
```

All comparisons use a small **zero-dropout FP32 model** and tolerate
normal floating-point rounding. Separate tests cover dropout and causal
masking. Requiring exact bitwise equality would incorrectly reject valid
optimized kernels.

Run `python -m pytest tests/test_sdpa_optimizer_parity.py -q`.

**Experiment:** Tighten the absolute tolerance until one case fails.

**Checkpoint:** Why might matching logits be insufficient to guarantee
the next optimizer update also matches?
