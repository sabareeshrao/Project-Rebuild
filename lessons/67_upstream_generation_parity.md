# Lesson 67: compare generated tokens against original nanoGPT

Matching single-step logits is necessary but not sufficient for
matching **autoregressive sampling**. A difference in context
cropping, temperature, top-k cutoff or RNG state can change
every subsequent token.

This optional test loads a reviewed original nanoGPT `model.py`,
transfers our fused GPT weights to it, then generates with the
same prompt and sampling settings:

```text
our GPT + matching weights --+--> seed 33 -> sample N tokens
original GPT + same weights -+--> seed 33 -> sample N tokens
                                      |
                                assert exact ID equality
```

The helper preserves the caller's CPU random-number state and the
original train/eval modes. The test exercises a prompt longer than
`block_size`, three temperature/top-k combinations, and two
batch rows. It is explicitly CPU-only.

Run against the same reviewed pinned upstream file as Lesson 62:

```bash
NANOGPT_MODEL_PY=original_nanogpt_model.py python -m pytest tests/test_actual_nanogpt_generation.py -v
python examples/64_compare_upstream_sampling.py --upstream-model original_nanogpt_model.py
```

The optional integration tests skip when the upstream file isn't
provided. A separate unit test uses a clearly identified local test
double only to verify that the parity harness detects mismatches
and restores RNG/modes.

**Checkpoint:** Why must two models start from the same input
*and the same RNG state* to compare random generation exactly?

**Limit:** This is synthetic-weight CPU parity, not a verification
that pretrained GPT-2 files or upstream training checkpoints load.
