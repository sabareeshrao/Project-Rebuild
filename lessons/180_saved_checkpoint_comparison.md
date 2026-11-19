# Lesson 180 — Compare two saved GPT inference checkpoints

The previous paired validation exercises compared model objects
already held in memory. Real experiments save model weights to disk
and we often want to compare two runs *after they finish*.

`compare_saved_checkpoints` loads two self-contained inference
checkpoints, including tokenizer metadata, then evaluates both
models on the same held-out windows.

```python
result = compare_saved_checkpoints(
    "out/model_before.pt", "out/model_after.pt",
    "data/tiny_story.txt", seeds=(1,2,3),
    batch_size=4, block_size=8,
)
```

A critical safety check compares the complete **character-to-ID
vocabulary order**. Matching only `vocab_size` is insufficient:
two character models could assign ID 2 to different characters.
Such a comparison would be meaningless.

The loader uses the existing strict, weights-only inference checkpoint
format and restores global CPU RNG state after model reconstruction.
It compares held-out text with the same tokenizer and contiguous 20%
validation split used by the training CLI.

The result includes model configurations, per-batch losses and
paired deltas, along with descriptive improved/worsened counts.
These samples are not statistically independent, and the sign
counts are not a significance test.

Run `python -m pytest tests/test_checkpoint_comparison.py -q`.

**Experiment:** Save the same model twice, then confirm all paired
deltas are exactly zero.

**Checkpoint:** Why is comparing tokenizer vocabulary *order* just
as important as comparing model output shapes?
