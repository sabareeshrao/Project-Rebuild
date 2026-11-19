# Lesson 268: diagnose which GPT restart points remain usable

A missing intermediate checkpoint is not necessarily fatal: if an
earlier audited checkpoint is intact, the optimizer can restart there.

The read-only inspector uses SHA-256 on the real files and labels
the run **ready**, **degraded**, or **unrecoverable**. It never loads
a PyTorch checkpoint and never changes any files.

```text
trusted audit + original corpus and reference
     |-> validate these core digests (hard failure if changed)
     +-> check intermediate stage hashes
               |-> all valid: ready
               |-> some valid: degraded; choose latest valid
               +-> none valid: unrecoverable
```

Run `python examples/161_recovery_health.py` after creating a
multi-stage audit with `examples/150_multistage_resume_audit.py`.

**Experiment:** In a disposable COPY of the run, delete the step-4
checkpoint. Why can step 2 still serve as a valid restart point?

**Checkpoint:** Why does a changed uninterrupted reference prevent
an exact-recovery parity claim?
