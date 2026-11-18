# Lesson 102 — Use mixed precision from the training CLI

The same text training command now accepts `--precision`:

```bash
python -m rebuild_gpt.train_cli --device cpu --precision fp32 --steps 2
python -m rebuild_gpt.train_cli --device cpu --precision bf16 --steps 2
# CUDA required; gradient scaling is automatic:
python -m rebuild_gpt.train_cli --device cuda --precision fp16 --steps 2
```

The FP32 default preserves previous behavior. The JSON history records
`requested_precision`. Resumable training-state controls also record
BF16/FP16 so mismatched checkpoints are rejected.

FP16 scaler state is saved and restored with optimizer state and RNG.
If overflow occurs, GradScaler may skip an update; our `--steps`
counts **attempted** optimizer iterations, including skipped ones.

Run locally:

```bash
python examples/87_mixed_precision.py --device cpu --precision bf16
python -m pytest tests/test_precision.py tests/test_amp_steps.py \
    tests/test_mixed_precision_training.py tests/test_precision_cli.py -q
```

**Experiment:** Compare BF16 and FP32 loss values on the same small
corpus and fixed random seed. Why do the values differ slightly?

**Checkpoint:** Which parts of training stay FP32 even when we use BF16
autocast? When and why is gradient scaling required?

**Limitations:** Actual GPU throughput, CUDA FP16 overflow recovery and
cross-machine reproducibility must be checked on supported hardware.
Distributed training remains outside this milestone.
