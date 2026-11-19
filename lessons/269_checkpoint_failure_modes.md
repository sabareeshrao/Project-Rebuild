# Lesson 269: missing, truncated, and appended-corruption checkpoints

The original fault drill appended marker bytes to a COPY of the
saved checkpoint. Real failures can also leave a truncated file
(e.g. interrupted write) or remove the file completely.

We now support three mutually exclusive modes:

- `append`: extra bytes invalidate the original file hash.
- `truncate`: half of the copied file's bytes are lost.
- `remove`: the copied checkpoint is missing.

All three must cause the latest stage's audit digest to fail, while
an earlier verified stage remains available for recovery. The
original artifacts are never changed.

Run `python examples/162_truncated_checkpoint_drill.py` after a
fresh multi-interruption audit. Choose a new output directory for
subsequent runs.

**Checkpoint:** Why can hash checking detect truncation without
attempting to deserialize the damaged PyTorch file?
