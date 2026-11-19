# Lesson 291 — Independently rerun correctness checks on trained weights

Our prior experiment generated a real training checkpoint and
conformance evidence, and then verified that the hashes corresponded
to the exact same files.

Now independently rerun the **eight model-level checks** and compare
their fresh measurements with the saved report.

```text
trained model + held-out corpus
          |
   hash-verified manifests
          |
   rerun all eight invariants
          |
   compare saved / fresh errors
          |
   reproduction.json (only after success)
```

Any failed recheck removes an old `reproduction.json` success marker.
The original training files and conformance report are not overwritten.

Run lesson 289, then
`python examples/184_reproduce_review.py`.

**Question:** Why does an independent rerun provide stronger evidence
than a file hash, while still falling short of proving GPT-2 quality?
