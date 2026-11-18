# Lesson 62: compare with the real original nanoGPT

The parity harness only proves compatibility if we use the
**actual upstream implementation**, not a second copy of our own code.

For reproducibility, use the upstream revision:

```text
karpathy/nanoGPT
3adf61e154c3fe3fca428ad6bc3818b27a3b8291
```

Obtain `model.py` from the official repository and inspect it before
running. For example, from the root of this project:

```bash
curl -L https://raw.githubusercontent.com/karpathy/nanoGPT/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py -o /tmp/original_nanogpt_model.py
NANOGPT_MODEL_PY=/tmp/original_nanogpt_model.py python -m pytest tests/test_actual_nanogpt_parity.py -v
python examples/59_compare_original_nanogpt.py --upstream-model /tmp/original_nanogpt_model.py
```

The test compares three small CPU configurations (bias on/off and
one/two transformer blocks). It transfers our exact fused model
weights into upstream GPT via named tensors and verifies:

- Full per-position logits when targets are supplied.
- Only last-position logits during inference.
- Masked next-token cross-entropy (ignore target `-1`).
- Matching tensor names/shapes and tied embeddings.

The integration test is **skipped** when `NANOGPT_MODEL_PY` is unset,
so normal offline test runs do not download or execute third-party
code. Passing these small tests is evidence of mathematical parity
for the exercised configurations, **not proof** that real large GPT-2
checkpoints or all upstream training workflows are compatible.

**Experiment:** Run with `--bias-free`, then remove one tensor from
the transferred state and observe why strict validation fails.

**Checkpoint:** What additional tests would be needed to verify
pretrained GPT-2 checkpoint loading and numerical parity on GPU?
