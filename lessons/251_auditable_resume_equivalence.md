# Lesson 251 — Record evidence, not just a passing console message

The controlled experiment verifies that training 4 steps uninterrupted
produces **identical meaningful state** to training 2 steps, saving,
reloading and training 2 more.

The audit JSON records exact source corpus and checkpoint hashes, the
training settings, verification results, Python/PyTorch versions and
validation metrics.

```text
full run checkpoint  ─┐
partial -> resumed   ─┴─> exact-state comparison -> passing audit
                                   |
                         failure -> no passing report
```

Checkpoint file hashes do **not** need to be identical because
serialization can differ in nonsemantic details. We check the actual
deserialized contents to establish equivalence. Hashes identify the
specific artifacts that were compared.

Run `python examples/144_resume_audit.py` to create the JSON audit.

**Checkpoint:** Why should a previously generated `passed` report
not remain in place after an attempted rerun fails?
