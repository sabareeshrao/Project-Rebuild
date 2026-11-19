# Lesson 263: read a recovery report critically

A JSON file can contain the string `"passed"` even if it was
partially written or manually edited. Before trusting its claims,
validate the expected format, both SHA-256 digests and the relationship
between the restart step, final step and comparison result.

    recovery_audit.json -> parse -> validate digests
                                  -> validate restart < final
                                  -> validate exact-state claim

This is **schema validation**, not an integrity or authenticity proof.
The next exercise recomputes hashes from the saved files and checks
the deserialized training state.

Run the earlier recovery command, then:
`python examples/156_read_recovery_audit.py`

**Experiment:** Change `target_step` in a COPY of the report without
changing the final comparison. Which consistency rule rejects it?
