# Lesson 231: make a parity claim auditable

Even a passing local-vs-upstream comparison is hard to reproduce
without knowing which upstream Python source was executed.

A SHA-256 checksum identifies the **exact bytes** of the upstream
`model.py`. The audit can reject an unexpected checksum BEFORE
importing any reference code.

```text
expected SHA-256? -> hash reviewed upstream source
                     | mismatch -> stop
                     v match
                forward parity + generation parity
                     | either fails -> no "passed" report
                     v both succeed
                JSON report with input file hashes,
                errors, matched tensor count and settings
```

Run:
```bash
python examples/124_capstone_parity_audit.py \
  --upstream-model /trusted/nanoGPT/model.py \
  --upstream-sha256 <actual-64-character-sha256>
```

Obtain the digest locally via `sha256sum /trusted/nanoGPT/model.py`
(or a trusted equivalent). Never execute a reference file you haven't
reviewed. The report records environment versions; it does not prove
model quality, training equivalence or generalization.

**Question:** Why is a successful test against an unpinned, mutable
reference insufficient for a long-lived reproducibility claim?
