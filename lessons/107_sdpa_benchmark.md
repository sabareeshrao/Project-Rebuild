# Lesson 107 — Benchmark SDPA vs explicit attention on your hardware

At this point we can train the text GPT with `attention_impl="sdpa"`
while retaining the original fused-manual implementation. The CLI
exposes `--attention-impl fused|sdpa`.

```bash
python -m rebuild_gpt.train_cli --steps 2 --attention-impl sdpa
python examples/88_attention_benchmark.py --iterations 10 --warmup 3
# On compatible NVIDIA hardware:
python examples/88_attention_benchmark.py --device cuda --precision bf16
```

The benchmark creates two equal-sized models with **identical weights**,
measures inference median latency after warmup, synchronizes CUDA
kernels around measurements, and reports the maximum absolute difference
between their logits.

```text
same weights, same inputs
     ├─ explicit fused attention → median milliseconds
     └─ SDPA attention            → median milliseconds
                 ↳ ratio and max numerical difference
```

PyTorch chooses which SDPA kernel to run, and SDPA is **not guaranteed**
to beat manual attention for every device, context length, shape, or
precision. Tiny CPU inputs may favor the simpler path.

Run `python -m pytest tests/test_sdpa_training.py tests/test_attention_benchmark.py -q`.

We also add a CPU GitHub Actions workflow. CUDA tests skip on runners
without a GPU; this is not proof of GPU performance or correctness.

**Experiment:** Increase context length while holding other dimensions
fixed. Compare the ratio of SDPA vs manual latency.

**Checkpoint:** Why must we synchronize CUDA before/after timing, and
why is a single timing sample less meaningful than the median?
