# Lesson 160 — Opt in to cached evaluations during GPT training

Training repeatedly checks whether the current GPT improves on a
fixed validation sample. The existing evaluator regenerates the
same CPU batches from the same seed each time. We can avoid repeated
sampling while keeping the *mathematics identical*.

```python
train_char_gpt(
    "data/tiny_story.txt",
    steps=5, eval_interval=1, eval_batches=4,
    eval_batch_backend="cached",
)
```

The trainer builds one CPU batch plan for training-loss estimation
and one for validation-loss estimation. They use independent seeds
`seed + 2` and `seed + 3`, just like the existing implementation.

The option applies only to **sampled validation** with an evaluation
interval. The default backend remains `regenerate`; no historical
behavior changes unless caching is requested. Caching does not change
training batches or optimizer settings.

We record the nondefault evaluation backend in resumable training
controls, rejecting restarts that switch it. Unit tests compare
*every recorded validation result*, final weights and checkpointed
continuation against the original implementation.

Run `python -m pytest tests/test_cached_eval_training.py -q`.

**Experiment:** Compare two three-step training runs, one with
cached evaluation and one without, using identical seeds.

**Checkpoint:** Why can caching evaluation inputs be harmless for the
optimizer trajectory when caching training inputs would not be?
