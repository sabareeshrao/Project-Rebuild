# Lesson 285 — Recompute the evidence, not just the file hashes

Checking a SHA-256 establishes that the saved report refers to the
same checkpoint and corpus. It does **not** establish that the
eight measurements inside the report were computed correctly.

We can do better by independently rerunning the checks and comparing
the new results against the stored ones:

```text
trusted report --(validate + verify hashes)--+
                                              |
exact model + corpus -> run eight checks again
                                              |
                   compare recorded and fresh measurements
```

Run the original audit in lesson 282, then
`python examples/178_reproduce_conformance.py`.

Small floating-point tolerance accommodates rounding differences,
while model settings, seed and SHA-256 digests must agree exactly.

**Exercise:** Change a numeric error field in a copy of the JSON
without altering the model. Why will a fresh run reject this?

Reproduction verifies these **specific** CPU checks, not the model's
ability to generalize or match all of upstream nanoGPT.
