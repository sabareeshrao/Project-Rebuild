# Lesson 257: make multi-restart training evidence auditable

The final exercise compares six uninterrupted optimizer steps to the
same six steps interrupted twice. Every stage uses the existing,
tested training/resume implementation and retains optimizer state.

    uninterrupted: 1 2 3 4 5 6 -> uninterrupted.pt
    restarted:     1 2 | 3 4 | 5 6 -> multistage_final.pt

We inspect each saved intermediate state for the correct step,
corpus identity, model configuration, optimizer controls and
evaluation-history prefix. The final states must agree exactly.

Only after all checks pass does the command publish a JSON audit
containing relevant file SHA-256 digests and runtime versions.
A failed rerun never leaves an old passed audit in its place.

Run:

    python examples/150_multistage_resume_audit.py --stops 2,4

Experiment: Try --stops 4,2. Why should validation reject this
before any optimizer update occurs?

Checkpoint: Which changes would be required to test reproducibility
across mixed precision, stochastic dropout, or distributed training?
