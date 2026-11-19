# Lesson 148 — Separate sampling, transfer and optimizer costs

A training-step latency number mixes several distinct operations:

```text
CPU token stream → sample batch → transfer/stage → forward/backward/AdamW
                        |              |                  |
                    sample ms      transfer ms         update ms
```

We now have a timing helper that records all three stages and
end-to-end duration after a controlled warmup.

For CUDA, we **synchronize after every stage**. That prevents
asynchronous kernel launches from being mistakenly charged to the next
stage, but also deliberately removes any possible transfer/compute
overlap. A sum of segmented times is **not** a speed prediction for a
future overlapping implementation.

The same utility works on CPU without initializing CUDA. Each result
contains raw samples and median/min/max latencies. The slowest stage
is the one with the highest median.

Run `python -m pytest tests/test_pipeline_timing.py -q`.

**Experiment:** Compare the stage medians of a data-heavy workload
to a model-heavy workload.

**Checkpoint:** Why does synchronizing each stage improve attribution
but potentially misrepresent what an overlapping pipeline could do?
