# Lesson 106 — Swap attention kernels in a complete GPT

The original architecture now offers **three readable configuration
choices**:

- `attention_impl="educational"`: separate heads, for intuition.
- `attention_impl="fused"`: packed QKV, explicit attention matrix.
- `attention_impl="sdpa"`: the same packed QKV weights, but attention
  calculated by PyTorch's scaled-dot-product attention function.

```text
GPT tokens → embeddings → transformer block(s)
                             |
                 packed Q,K,V (identical weights)
                      /             \
            manual attention     PyTorch SDPA
                      \             /
                      residual + MLP → vocab logits
```

The config is saved in checkpoints, so loading an SDPA model selects
the same computation path. The *tensor names and shapes* for fused
and SDPA models are identical, enabling strict weight transfer.

Run `python -m pytest tests/test_sdpa_gpt.py -q`.

**Experiment:** Save a state dict from `attention_impl="fused"` and
strictly load into the same-size `attention_impl="sdpa"` model.

**Checkpoint:** What remains exactly the same when switching kernels,
and what might differ slightly because of floating-point execution?
