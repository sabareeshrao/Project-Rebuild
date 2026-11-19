# Lesson 165 — Apply a CPU cache budget to resumable GPT training

The character-GPT trainer now accepts an optional MiB limit when
using `eval_batch_backend="cached"`:

```python
train_char_gpt(
    "data/tiny_story.txt",
    steps=5, eval_interval=1, eval_batches=4,
    eval_batch_backend="cached", eval_cache_max_mib=16,
)
```

The trainer first calculates the combined conservative storage for
**both** evaluation splits. If it exceeds the limit, the run fails
explicitly before creating either batch plan. The limit is *not*
applied to the complete training process or GPU VRAM: it only bounds
the logical int64 tensors retained by sampled evaluation plans.

The cap is opt-in. Earlier cached and regenerated training runs
behave as before when it is absent.

The nondefault budget is recorded in resumable training controls;
restarting with a different cap is rejected, as is switching between
cached and regenerated evaluation. Identical seeds and sufficient
budgets still produce identical evaluation histories and weights.

Run `python -m pytest tests/test_eval_cache_training_budget.py -q`.

**Experiment:** Set a cap that allows one split but not both. Verify
that no evaluation batches are prepared before the error.

**Checkpoint:** Why must a training checkpoint record its evaluation
cache policy even if the optimizer objective is unchanged?
