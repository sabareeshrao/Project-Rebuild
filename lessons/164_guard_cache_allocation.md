# Lesson 164 — Reject oversized evaluation caches before allocation

A memory budget is useful only if it is checked *before* sampling
the validation token windows.

`prepare_evaluation_batches(..., max_bytes=...)` now bounds one
split. More importantly, the helper
`prepare_train_val_evaluation_batches` checks a **combined limit**
for the training-evaluation and validation-evaluation batches before
sampling either of them.

```text
estimate both splits: B × T × K × 2 targets × 2 splits × 8 bytes
                           ↓
                 fits configured budget?
                        /          \
                       yes          no
                        |            |
                  sample both     raise error
```

When the budget is exceeded, the helper raises an explicit
`ValueError`. It never silently reduces `eval_batches` or changes
the seed, because either would alter the experiment being resumed.

The existing `prepare_evaluation_batches` API remains compatible:
`max_bytes=None` means no explicit cap, and sample order is unchanged.

Run `python -m pytest tests/test_eval_cache_limits.py -q`.

**Experiment:** Set a limit large enough for one split but not for
both splits. Check that zero batches are sampled.

**Checkpoint:** Why is a combined preallocation check safer than
first caching training examples and then discovering that validation
would exceed the remaining memory budget?
