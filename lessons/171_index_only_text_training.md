# Lesson 171 — Index-only evaluation in reproducible text training

The text trainer now supports a third sampled evaluation backend:

```python
train_char_gpt(
    "data/tiny_story.txt", steps=5, eval_interval=1,
    eval_batch_backend="indices",
)
```

The original `regenerate` backend redraws a fixed RNG sequence at
each validation; `cached` stores the resulting CPU token batches;
the new `indices` backend stores only their starting positions.

All three evaluate exactly the same token IDs and the current model
weights. Tests compare histories, metrics and final model parameters
under indexed/cached training sampling, microbatching and gradient
accumulation.

An optional `eval_cache_max_mib` applies to either stored backend,
using the appropriate logical storage estimate. The index backend's
limit counts retained start vectors, **not the shared corpus** or
temporary windows gathered during evaluation.

Changing backends or memory caps across a resume is rejected using
the existing checkpoint controls. The default remains `regenerate`,
so earlier checkpoints retain their behavior.

Run `python -m pytest tests/test_index_eval_training.py -q`.

**Experiment:** Set a cap that fits the compact index plan but is
smaller than the full cached batch estimate. Why does only one fit?

**Checkpoint:** What extra CPU work must index-only evaluation do at
each validation event compared with fully cached token windows?
