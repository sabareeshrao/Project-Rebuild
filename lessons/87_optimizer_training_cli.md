# Lesson 87: combine gradient accumulation with nanoGPT-style AdamW

The CLI now provides an opt-in path resembling more of original
nanoGPT's optimizer configuration:

```bash
python -m rebuild_gpt.train_cli \
  --corpus data/tiny_story.txt --steps 30 \
  --grouped-adamw --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.95 \
  --gradient-accumulation-steps 3 \
  --grad-clip 1.0 \
  --warmup-steps 3 --lr-decay-steps 25 --min-lr 0.002 \
  --eval-every 10 \
  --output out/char_gpt.pt \
  --best-output out/best_char_gpt.pt \
  --history-file out/training_history.json
```

The trainer now samples three microbatches for **each optimizer
update** and averages their gradients. It clips only after all three
backward passes. AdamW applies decay to 2D+ parameters, not biases
or LayerNorm, and warmup/cosine decay advances per optimizer step.

The JSON history captures accumulation steps, grouped-optimizer flag,
decay, beta values, evaluation metrics, and model configuration.
Checkpoints continue to include the original character vocabulary
and support later text generation.

Run: `python examples/84_optimizer_training_cli.py`.

**Experiment:** Repeat training with accumulation 1 vs 3 while
holding the *optimizer step count* fixed. Compare wall time,
validation loss, and selected checkpoint.

**Checkpoint:** Why do accumulating three microbatches and making
one AdamW step differ from making three consecutive AdamW steps?

**Limitations:** This CPU-oriented version does not yet implement
mixed precision, distributed data parallelism, or optimizer-state
resume inside the new text-training CLI. Those belong to later lessons.
