# Lesson 61: compare with the actual reference model

After mapping names and parameter shapes, our next question is whether
the **same tensors** cause the same **predictions** in both models.

`upstream_parity.py` imports a trusted, locally supplied copy of the
original nanoGPT `model.py`. No upstream implementation is copied
into this educational repository.

The helper builds its original `GPTConfig`, creates the original GPT,
and copies our fused model's tensors by their mapped names. It rejects
unknown keys and incompatible shapes; the only permitted extra keys
are older PyTorch causal-mask buffers. It then compares:

```text
full training logits  [B,T,V]
inference logits      [B,1,V]
cross-entropy loss, including ignored target -1
```

The two models are put into evaluation mode, to remove dropout noise.
Matching is numerical (with small floating-point tolerances), not a
demand for bit-identical low-level operations.

We deliberately **do not** download or execute unreviewed code from
inside the library. Loading source executes Python, so only supply
a trusted upstream file.

**Checkpoint:** Why must weights be copied before comparing logits,
rather than merely using the same random seed for two different models?
