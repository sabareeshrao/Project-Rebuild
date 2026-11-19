# Lesson 126 — Selective checkpointing in the text trainer

The GPT training function and CLI can now checkpoint every Nth
transformer block while retaining the earlier on/off behavior.

```bash
# Every block (previous behavior):
python -m rebuild_gpt.train_cli --activation-checkpointing --steps 2

# Only every second block (new behavior):
python -m rebuild_gpt.train_cli --n-layer 4 \
  --checkpoint-every-n 2 --steps 2
```

The options are intentionally **mutually exclusive**. Setting
`checkpoint_every_n=2` selects block indices `(1, 3)` in a
four-block model. An interval greater than the number of blocks
is rejected rather than silently doing nothing.

The enabled interval is saved in the **training-state controls**. A
resume with a different recomputation policy fails the compatibility
check. Previous runs with the default policy retain compatible
metadata, and inference checkpoint weights remain unchanged.

The JSON training report records both the previous boolean flag
and the optional interval to make experiments auditable.

Run: `python -m pytest tests/test_selective_training_cli.py -q`.

**Experiment:** Resume a run with `--checkpoint-every-n 2` using
`--checkpoint-every-n 1`. Observe the deliberate mismatch error.

**Checkpoint:** Why must a resume preserve the training policy,
even if recomputation should usually yield equivalent mathematics?
