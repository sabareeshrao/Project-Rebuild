# Lesson 155 — A drop-in sampler backend for real training

The character-GPT trainer now supports an opt-in
`sampling_backend="cached"` option:

```bash
python -m rebuild_gpt.train_cli --steps 3 --sampling-backend cached
```

The **default remains indexed sampling**. When opted in, the
trainer builds one `CachedWindowSampler` over the training token
stream. Its `unfold` table is only a view, and random starting
positions use the same `training_rng` as the original implementation.

Tests compare parameter tensors and validation histories after
multiple updates, including logical microbatching and independent
gradient accumulation. They also verify resume behavior.

```text
same seed + training RNG
   ├─ indexed gather → batch (x, y)
   └─ cached unfold  → batch (x, y)
                            ↓
                     same GPT update
```

The backend is recorded in resumable training-state controls **when
enabled**. Switching during a restart fails the identity check,
while older default-indexed states remain compatible. The CLI
history JSON records the selected policy.

Run `python -m pytest tests/test_cached_text_training.py -q`.

**Experiment:** Compare model weights after three updates from
the same seed using indexed vs cached sampling.

**Checkpoint:** Why must changing a data-loading implementation
preserve the *entire sequence* of RNG draws, not just one batch?
