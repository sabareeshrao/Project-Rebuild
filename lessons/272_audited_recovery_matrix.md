# Lesson 272: audit all independent checkpoint recovery outcomes

One successful recovery does not establish that every common failure
mode can be handled. Our matrix separately performs:

```text
trusted original audited GPT run
   |--> append bytes (copy)   -> restart -> exact comparison
   |--> truncate file (copy)  -> restart -> exact comparison
   +--> remove file (copy)    -> restart -> exact comparison
                                        |
                     SHA-256 audit ONLY if every case passed
```

Run after generating a multi-stage training audit:

```bash
python examples/150_multistage_resume_audit.py
python examples/165_recovery_matrix_audit.py
```

The result lists each fault mode, fallback step, SHA-256 of the
recovered weights and SHA-256 of its own recovery audit. Successful
state comparisons happen before the aggregate audit is written.

The matrix refuses to reuse existing output directories and never
modifies the original training checkpoints.

**Experiment:** Run with `--modes truncate,remove` and a fresh
directory. How does the reported number of trials change?

**Checkpoint:** Why does this give evidence of recovery behavior
under chosen faults, but not a guarantee against arbitrary storage
failures or attacks on the trusted manifest?
