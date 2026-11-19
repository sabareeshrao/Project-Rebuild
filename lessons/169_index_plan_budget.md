# Lesson 169 — Enforce a much smaller memory limit for index plans

A full cached evaluation retains sampled token contexts and targets.
An index plan needs only the randomly chosen **starting positions**
and a reference to the original CPU corpus.

For two splits, K batches, B starts per batch and 64-bit indices:

```text
index bytes = 2 × K × B × 8
```

We now calculate this separately using `estimate_index_plan_bytes`.
A combined `prepare_train_val_index_plans` validates *both* corpus
streams and the byte budget before drawing any random starts.

The index-only calculation excludes the shared token corpus, which
must remain in memory, and excludes temporary tensors created when
we later gather windows for a model forward pass. This is a plan
storage budget, **not total RAM usage**.

A memory guard rejects an over-budget plan rather than silently
reducing validation samples, preserving experiment comparability.

Run `python -m pytest tests/test_eval_index_budget.py -q`.

**Experiment:** Calculate bytes required for B=64, K=32 and
compare an index plan with full cached token windows of T=128.

**Checkpoint:** Why must the budget cover both train and validation
plans before the first set of starts is drawn?
