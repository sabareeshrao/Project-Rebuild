# Lesson 282: eight ways to test a saved GPT without upstream code

A successful language-model training run does not by itself verify
that all the details of the implementation are correct.

The first four checks (lessons 273–277) cover causality, batch
isolation, forward/loss semantics and an independent numerical
gradient.

This phase adds tied-weight identity, private-seed generation,
checkpoint round-tripping, and manual-vs-SDPA attention parity.

```text
checkpoint + held-out corpus
   |--> causal prefix, batch isolation, forward/loss, gradient
   |--> tied weights, seeded sample, save/load, SDPA vs manual
   +--> eight validated results -> JSON report (only if all pass)
```

Run after saving a small checkpoint:

```bash
python examples/120_capstone_cli.py --steps 4
python examples/175_extended_conformance.py
```

The audit records exact input file hashes and the maximum numerical
errors from applicable comparisons. The old four-check CLI remains
available for backward compatibility.

**Question:** Which additional evidence would you require before
claiming parity with original nanoGPT's full training loop or a
GPT-2-quality language model?
