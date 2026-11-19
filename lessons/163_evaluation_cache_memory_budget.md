# Lesson 163 — Budget the memory cost of cached evaluations

A prepared evaluation cache speeds up repeated token sampling by
retaining fixed training- and validation-evaluation examples in CPU
memory. We should estimate that cost before allocating tensors.

For batch B, context T, evaluation batches K, and S independent
splits, a conservative token-storage budget is:

```text
logical bytes = 2 × B × T × K × S × 8
                      ↑               ↑
                 x and y          int64 token IDs
```

The current batch sampler produces inputs and targets that share an
underlying `[B,T+1]` allocation. Accordingly, this equation counts
logical tensor elements and is intentionally a **conservative bound**,
not physical process RSS. Other allocations also contribute to RSS.

`estimate_evaluation_cache_budget` returns both a per-split estimate
and the combined total. `enforce_evaluation_cache_budget` rejects a
plan above an explicit byte limit without silently changing its
sampling policy.

Run `python -m pytest tests/test_eval_cache_budget.py -q`.

**Experiment:** Increase `eval_batches` fourfold. How does the
conservative storage budget change?

**Checkpoint:** Why should a memory guard run *before* generating
random indices or constructing any cached evaluation tensors?
