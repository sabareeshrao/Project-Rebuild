# Lesson 293 — The 300-commit milestone: follow the evidence

This is the **300th planned learning commit** of an original
nanoGPT-inspired rebuild. The number itself is not a benchmark score.

A complete local review has three independent claims:

1. The tiny GPT actually trained and produced a checkpoint and sample.
2. Eight model-level conformance checks used **that exact checkpoint**.
3. An independent rerun checked the saved measurements again.

Our read-only verifier now connects the three manifests by SHA-256:

```text
capstone/run.json -- model and corpus hashes --+
extended_conformance.json ---------------------+-> same files
reproduction.json -- source report hash -------+-> eight rerun checks
```

Run the three `release_review_cli` modes from lesson 292 first, then
`python examples/186_release_evidence.py`.

This check does **not** load `.pt` files or re-execute training. It
verifies the reports' relationships. Source manifests must still be
trusted; hashes alone are not signatures.

**Question:** What additional evidence would you need to claim the
implementation reproduces original nanoGPT's full GPU training loop?
