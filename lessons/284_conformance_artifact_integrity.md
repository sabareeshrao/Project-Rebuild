# Lesson 284 — A saved report must match the actual files

After validating the report's JSON fields, recompute the hashes of the
*real* model checkpoint and corpus.

```text
trusted conformance report -> recorded SHA-256
saved model checkpoint ----> recomputed SHA-256 -> compare
original text corpus ------> recomputed SHA-256 -> compare
```

An altered or missing checkpoint cannot legitimately inherit the
passing report from an earlier model. We check that without loading
any PyTorch file or executing Python code from a checkpoint.

Run `python examples/177_verify_conformance_files.py` after lesson
282 generates the saved JSON report.

**Exercise:** Append one byte to a COPY of the checkpoint file.
Why does this change its digest even if all its original bytes remain
present?

**Security note:** A manifest's hash proves file identity only
relative to the *trusted* manifest; it is not a signed attestation.
