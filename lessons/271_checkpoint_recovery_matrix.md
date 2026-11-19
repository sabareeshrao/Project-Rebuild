# Lesson 271: one recovery policy, three independent failure scenarios

A robust recovery experiment should not pass solely because the
selected fault was easy to detect. The matrix applies distinct
faults to **separate copies** of an audited run:

```text
original run (read-only)
    |-> append copy   -> detect -> fallback -> replay -> verify
    |-> truncate copy -> detect -> fallback -> replay -> verify
    +-> remove copy   -> detect -> fallback -> replay -> verify
```

Every trial must verify the recovered model parameters, AdamW moments,
RNG state and training history against the uninterrupted run.
If any trial fails, the matrix cannot report `passed`.

Run `python examples/164_recovery_matrix.py` after completing the
multistage audit. This deliberately performs multiple tiny CPU
training replays. It requires a new output directory.

**Checkpoint:** Why is each failure scenario placed in a distinct
directory instead of repeatedly corrupting one run directory?
