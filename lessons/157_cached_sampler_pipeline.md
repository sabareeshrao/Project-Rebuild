# Lesson 157 — Does a faster sampler matter for full GPT training?

A microbenchmark can identify CPU sampling overhead, but the entire
training step also includes transfers, forward and backward, AdamW,
and Python bookkeeping. A faster sampler may make little difference
if the model update dominates elapsed time.

We now expose `--sampling-backend indexed|cached` in both pipeline
profilers:

```bash
python examples/101_training_pipeline_breakdown.py \
  --sampling-backend indexed --csv out/pipeline_indexed.csv
python examples/101_training_pipeline_breakdown.py \
  --sampling-backend cached --csv out/pipeline_cached.csv

python examples/102_pipeline_scaling_sweep.py \
  --sampling-backend cached --batch-sizes 1,2,4 \
  --depths 1,2 --csv out/pipeline_cached_sweep.csv
```

The experiments use the same synthetic token stream, model
initialization, optimizer configuration, and sampling generator seed
for both choices. CSV and JSON record the backend, stage timing and
model workload. Existing scripts still default to the original
indexed backend.

```text
sampler latency → stage medians → end-to-end training duration
                           ↑
                 backward/AdamW may dominate
```

CUDA timing uses explicit synchronization after every stage and
cannot demonstrate transfer/compute overlap. CPU measurements make
no claims about GPU speed or VRAM. The tests verify comparable
workloads, not that caching must be faster.

Run `python -m pytest tests/test_cached_pipeline_profiles.py -q`.

**Experiment:** Compare sampler-stage and total-step latency for a
one-layer GPT and a four-layer GPT. Does sampler optimization matter
equally for both?

**Checkpoint:** Why can a large percentage reduction in sampling
latency produce only a small reduction in total training time?
