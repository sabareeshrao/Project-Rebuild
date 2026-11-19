# Lesson 168 — Cache the evaluation indices, not the token batches

A fixed sampled evaluation selects *starting positions* from the
training and validation streams. The same positions are reused on
every evaluation step, so we can store only those positions instead
of retaining complete next-token input/target tensors.

For B sequences, context T, and K evaluation batches:

- Materialized input/target logical bytes: `2 × B × T × K × 8`.
- Retained int64 start-index bytes: `B × K × 8`.

When T=32, the start-index plan retains substantially fewer bytes.
It still holds a reference to the original CPU corpus tensor, which
must remain alive. Reconstructing windows costs indexing work later.

`prepare_evaluation_index_plan` uses exactly the same seed-driven
sequence of `torch.randint` calls as the original evaluator.
Repeated calls to `plan.batches()` regenerate identical token pairs
without advancing global or training RNG.

Run `python -m pytest tests/test_eval_index_plan.py -q`.

**Experiment:** Increase the context length while keeping B and K
constant. How does index memory compare with cached batch memory?

**Checkpoint:** What work are we trading away when we retain only
random starts rather than already-gathered token batches?
