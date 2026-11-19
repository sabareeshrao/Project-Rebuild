# Lesson 290 — Did our tests inspect the actual trained checkpoint?

A successful training run and a successful conformance test can refer
to **different** checkpoints without anyone noticing. That is not
valid end-to-end evidence.

Our verifier reads two manifests from a single review directory and
compares them to the real model and corpus bytes:

```text
capstone/run.json --+  model SHA-256
                    +  corpus SHA-256 -> agreement required
extended_conformance.json --+
                               |
                     model configuration consistency
                     training validation metrics sanity
```

No checkpoint deserialization or model execution is needed to check
this identity relationship.

Run `python examples/182_review_capstone.py` and then
`python examples/183_verify_review_evidence.py`.

**Question:** Why is it insufficient for training and conformance
reports to be individually valid if they describe different weights?
