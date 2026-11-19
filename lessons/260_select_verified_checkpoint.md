# Lesson 260: identify the last good restart point

If an interruption checkpoint is lost or damaged, don't simply load
the newest .pt file. Match its exact SHA-256 against the known-good
audit first.

    stage 2 [verified] ----+
    stage 4 [damaged]  ----+-> choose stage 2
    stage 6 [final]    not an intermediate recovery point

The selector never loads or executes PyTorch checkpoint content. It
rejects a changed corpus and skips missing or modified stages. If
none are intact, it refuses to invent a recovery point.

Run `python examples/153_find_restart.py` after creating the
multi-restart audit in lesson 257.

**Experiment:** Delete only a COPY of the step-4 checkpoint. Why
does the selector choose step 2 instead of step 6?

**Security:** SHA-256 comparisons depend on a trusted manifest.
A matching hash identifies a file, not whether the file was
produced by an honest trainer or is otherwise safe to load.
