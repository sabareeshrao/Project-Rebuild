# Lesson 266: recover from a real corrupted checkpoint

This exercise proves the recovery workflow instead of merely
asserting that it *should* work.

  1. Copy an intact, audited multi-interruption run into a new directory.
  2. Corrupt the **latest intermediate** checkpoint in that copy.
  3. Select the previous hash-verified checkpoint.
  4. Restore the saved AdamW moments and random streams.
  5. Resume to the original final step.
  6. Independently verify recovered checkpoint bytes and complete state.

Run `python examples/150_multistage_resume_audit.py`, then
`python examples/159_run_recovery_drill.py`.

This intentionally refuses a plan with no earlier fallback.
Existing drill output directories are never overwritten; choose a
fresh location for each repeat.

**Checkpoint:** Why does the recovered run have to use the original
learning-rate schedule, not one restarted from step zero?

**Caution:** Only run this against your own trusted local training
checkpoints; the drill never modifies the original files.
