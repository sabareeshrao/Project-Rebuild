# Lesson 141 — Measure streaming transfers against eager transfers

The original microbatch performance script already measured complete
training steps, but it held its fixed synthetic batch on the same
device as the model. That obscured the cost of GPU-resident inputs.

Now the logical batch stays on **CPU** for both comparisons:

- `--transfer-mode eager`: move the whole logical batch to GPU, then
  split it into microbatches for backward.
- `--transfer-mode streamed`: move only the current microbatch, finish
  backward, and release it before moving the next chunk.

```bash
python examples/97_microbatch_benchmark.py \
  --device cuda --logical-batch-size 6 --microbatch-sizes 1,2,6 \
  --transfer-mode eager --csv out/micro_eager.csv

python examples/97_microbatch_benchmark.py \
  --device cuda --logical-batch-size 6 --microbatch-sizes 1,2,6 \
  --transfer-mode streamed --csv out/micro_streamed.csv
```

The workload is equal: same model shape, starting seed, logical token
IDs, optimizer update count, and warmup settings. The measured region
includes device transfer time and model forward/backward.

CUDA results show extra allocated bytes above an initialized-optimizer
baseline; this is **not total VRAM**. On CPU, both transfer modes
remain valid control paths, but GPU memory is reported as unavailable.

Run `python -m pytest tests/test_streamed_microbatch_benchmark.py -q`.

**Experiment:** Compare B=6, M=1 and M=6 in both transfer modes.

**Checkpoint:** Why may streaming input transfers add latency even
if they lower peak accelerator allocation?
