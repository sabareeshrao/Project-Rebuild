# Lesson 127 — Benchmark the selective recomputation tradeoff

We can now compare three activation policies using identical model
initialization, architecture, token IDs, and optimizer settings:

- **off**: regular forward/backward; store the usual activations.
- **every 2nd**: checkpoint only blocks 2, 4, 6, ... (1-based).
- **all**: checkpoint every transformer block.

```text
       off         every 2nd        all
     0 1 2 3      0 [1] 2 [3]     [0][1][2][3]
       |              |               |
    step time       step time       step time
     CUDA peak       CUDA peak       CUDA peak
```

Brackets indicate blocks whose forward activations are recomputed
during backward. The model parameter count stays identical.

Run:

```bash
python examples/94_selective_checkpoint_benchmark.py \
  --device cpu --n-layer 4 --checkpoint-every-n 2

# Requires a supported NVIDIA CUDA GPU:
python examples/94_selective_checkpoint_benchmark.py \
  --device cuda --precision bf16 --n-layer 4 --checkpoint-every-n 2
```

The script measures median full training-step latency with warmup
and, on CUDA only, additional PyTorch allocator peak bytes above a
baseline established after optimizer state has been initialized.

**Important:** Neither shorter latency nor lower VRAM is guaranteed
on every model. Small models are often dominated by fixed overhead.
Peak CUDA allocator delta is not total GPU memory; compare the same
machine under repeatable conditions.

Run `python -m pytest tests/test_selective_checkpoint_benchmark.py -q`.

**Experiment:** Change the interval from 2 to 3 and compare
which blocks were selected and the timing ratio.

**Checkpoint:** Why might a selective policy outperform the
all-checkpoint policy *in wall time* while using more GPU memory?
