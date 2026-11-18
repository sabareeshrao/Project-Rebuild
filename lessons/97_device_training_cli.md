# Lesson 97 — Choose CPU or CUDA from the training CLI

You can now run the same character-level GPT training code on either
CPU or an available NVIDIA CUDA device.

```bash
python -m rebuild_gpt.train_cli --device cpu --steps 2
python -m rebuild_gpt.train_cli --device auto --steps 2
# Only on a CUDA machine:
python -m rebuild_gpt.train_cli --device cuda --steps 2
```

The default is CPU for backward compatibility. `auto` selects CUDA
when PyTorch reports it available. Explicit CUDA fails clearly if no
compatible accelerator is found.

```bash
python examples/86_device_training.py --device cpu
python -m pytest tests/test_devices.py tests/test_evaluation.py \
  tests/test_text_training_devices.py tests/test_device_cli.py -q
```

Training-state checkpoints on CUDA now include CUDA RNG states. Inference
checkpoints still load on CPU. GPU results need not match CPU bit for bit
because floating-point kernel implementations differ.

**Experiment:** Compare CPU and CUDA training time and loss curves for a
tiny text model. Is GPU always faster on very small batches?

**Checkpoint:** What is the difference between choosing a computation
device and enabling mixed precision? (Mixed precision is a later lesson.)
