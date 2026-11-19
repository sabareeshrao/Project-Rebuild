# Lesson 267: one workflow for independent recovery verification

The recovery lab offers two operations with different effects.

**check:** Recheck a previously saved recovery audit, hash the
original corpus and checkpoints, then compare the recovered model,
optimizer moments and RNG state with an uninterrupted reference.
This does not perform more training.

**drill:** Copy the original audited run, corrupt an intermediate
checkpoint only in that new copy, recover from an earlier intact
checkpoint, and independently verify the final state.

```bash
python examples/150_multistage_resume_audit.py

python examples/160_recovery_lab_cli.py --mode drill \
  --source-dir out/multistage_resume \
  --drill-dir out/recovery_drill --damage-step 4

python examples/160_recovery_lab_cli.py --mode check \
  --source-dir out/recovery_drill
```

The drill destination must be new. To repeat, choose another
destination rather than overwriting previously saved evidence.

**Checkpoint:** A successful SHA-256 comparison cannot establish
who created a checkpoint. Why are trusted inputs and provenance
still needed in production systems?
