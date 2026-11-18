# Lesson 56: translate state-dict keys toward upstream nanoGPT

A model's saved tensors depend on both *mathematics* and its module
hierarchy. The original nanoGPT uses names such as:

```text
transformer.wte.weight
transformer.wpe.weight
transformer.h.0.ln_1.weight
transformer.h.0.attn.c_attn.weight
transformer.h.0.attn.c_proj.weight
transformer.h.0.mlp.c_fc.weight
transformer.h.0.mlp.c_proj.weight
transformer.ln_f.weight
lm_head.weight
```

Our fused model represents the same tensor roles under educational
names like `tokens.lookup.weight` and
`blocks.blocks.0.attention.qkv.c_attn.weight`.

`export_nanogpt_state_dict` converts **names and tensors** without
pretending to produce the original training checkpoint wrapper.
`load_nanogpt_state_dict` validates exact key sets, tensor shapes,
and consistency of the tied token/output embedding matrix.

This translation requires `attention_impl="fused"` and
`tie_embeddings=True`. It is a step toward compatibility, not proof
that a real original nanoGPT checkpoint loads correctly.

Run: `python examples/56_nanogpt_keys.py`.

**Experiment:** Intentionally delete one exported tensor or change
its shape before importing. Inspect the error.

**Checkpoint:** Why is matching the name of a tensor not enough
without also matching its shape, mathematical role, and configuration?

Reference: https://github.com/karpathy/nanoGPT/blob/master/model.py
