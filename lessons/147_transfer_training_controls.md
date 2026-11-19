# Lesson 147 — Opt-in pinned-memory controls in the training CLI

Our character-GPT trainer now accepts two *optional* CUDA transfer
options when using a logical microbatch size:

```bash
python -m rebuild_gpt.train_cli --device cuda \
  --batch-size 8 --microbatch-size 2 --steps 2 \
  --pin-memory --non-blocking-transfer
```

The behavior is explicit:

- **Default CPU/FP32** remains unchanged.
- `--pin-memory` requires CUDA and `--microbatch-size`.
- `--non-blocking-transfer` also requires `--pin-memory`.
- Each logical batch is staged in pinned CPU memory once.
- One pinned microbatch is copied on the current CUDA stream before
  its forward/backward pass; we do **not** create an overlapping stream.

Both enabled flags are recorded in resumable training-state controls,
so restarting with different transfer settings is rejected. Disabled
flags remain absent from the training-state metadata to preserve
older default-mode checkpoint compatibility. The CLI history JSON
always records both flags for clarity.

Run `python -m pytest tests/test_transfer_training_cli.py -q`.

**Experiment:** On CUDA, compare validation losses and elapsed time
with pinned transfers disabled versus enabled.

**Checkpoint:** Why must new optional performance controls be saved
in a reproducible training experiment's metadata even if they do not
change the intended next-token objective?
