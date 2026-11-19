# Lesson 289 — Prove the entire training-to-checking path

Our rebuild includes an actual training loop, a tiny text checkpoint,
sampled generation, and eight independent model-level correctness
checks. This lesson connects them into **one real CPU experiment**.

```text
UTF-8 corpus -> train a tiny GPT -> saved model.pt
                                 -> sampled text + run.json
                                 -> replay sample and verify hashes
                                 -> check eight model invariants
                                 -> extended_conformance.json
```

The output directory must be **new**. This protects earlier runs and
makes every experiment's evidence easy to distinguish. The training
and conformance steps use seeds recorded in their manifests.

Run `python examples/182_review_capstone.py` from the repository root.
Use the existing training and conformance CLIs separately if you want
to inspect an individual stage.

**Checkpoint:** Why should passing eight behavioral invariants *not*
be described as matching original nanoGPT's entire training loop?
