# Lesson 286 — Correctness across different held-out windows

A single conformance run checks one deterministic draw of held-out
token windows. To test more contexts, repeat all eight checks using
different random seeds.

```text
ONE saved model + corpus
      |-> seed 11: eight checks -> pass
      +-> seed 29: eight checks -> pass
                     |
              successful sweep JSON
```

The source checkpoint and corpus must have identical SHA-256 digests
across every round. The model configuration must also match.
The **sampled tokens are not expected to be identical across seeds**;
only the correctness invariants must continue to hold.

Run `python examples/179_conformance_seed_sweep.py` after creating
the capstone checkpoint.

**Exercise:** Which error might be missed by checking only one
token context, but caught by a different seed or context window?

This test broadens coverage but does not prove performance on
unseen language distributions or long-context workloads.
