# Lesson 259: check audit claims against actual file bytes

A coherent audit still does not prove its checkpoint files are present
or unchanged. We now stream each file through SHA-256 and compare the
digest against the recorded audit, including the original corpus.

    audit JSON -> stage steps and digests
             +-> actual corpus / checkpoints
                      |
                  SHA-256 compare
                      |
               verified or rejected

The verifier does NOT instantiate a model, load any PyTorch checkpoint,
or perform optimizer steps. It detects corruption and accidental
substitution of the files in a previously completed experiment.

Run:
    python examples/150_multistage_resume_audit.py
    python examples/152_verify_resume_artifacts.py

**Experiment:** Edit a byte in a COPY of `after_step_0004.pt`.
The verifier should identify that exact stage.

**Security:** A hash inside an editable manifest is not a signed
attestation; you must trust the origin of the manifest and digest.
