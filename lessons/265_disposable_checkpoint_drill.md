# Lesson 265: safely simulate a damaged training checkpoint

Do not corrupt your only model checkpoint as an experiment.

Instead, copy a previously audited multi-restart run into a NEW
directory. Confirm the copied files match the trusted manifest.
Only then append deliberately invalid bytes to one INTERMEDIATE
checkpoint copy.

    original audited checkpoints ------> unchanged
               |
               +---> disposable copy -> corrupt step 4
                                          |
                                   SHA-256 mismatch

Run:
`python examples/150_multistage_resume_audit.py`
then `python examples/158_prepare_recovery_drill.py`.

The destination must not already exist, and it cannot be inside the
source run. This is a fault-injection exercise, not an experiment in
malicious file construction.

**Checkpoint:** Why does appending even one byte invalidate the SHA-256
record even though most original checkpoint bytes remain unchanged?
