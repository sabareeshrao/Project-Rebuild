# Lesson 237: make training-parity results reproducible

We now have four distinct checks:

1. Inference logits and next-token loss
2. Sampled token sequences
3. Named gradients from a backward pass
4. Updated parameters after one identical plain AdamW step

An explicit reference-source SHA-256 digest helps make comparisons
auditable. The script refuses an unexpected digest **before importing**
the trusted source. It writes a `passed` JSON report only after the
gradient and updated-weight checks succeed.

```bash
python examples/120_capstone_cli.py --steps 4
python examples/130_training_step_audit.py \
  --upstream-model /trusted/nanoGPT/model.py \
  --upstream-sha256 <actual-reference-digest>
```

No downloaded reference code is shipped or automatically imported.
Review any externally supplied Python before running it.

**Experiment:** Modify a byte in a copy of the upstream source, then
retry with the original SHA-256. The import should not execute.

**Checkpoint:** Distinguish one-step equivalence from reproducing
nanoGPT's optimizer configuration and full training trajectory.
