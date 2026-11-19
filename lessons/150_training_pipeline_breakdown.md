# Lesson 150 — Find the largest part of a complete training step

We can now measure the actual character-GPT-style training pipeline,
instead of profiling sampling, device transfer and model updates in
unrelated processes.

```bash
python examples/101_training_pipeline_breakdown.py \
  --device cpu --batch-size 4 --block-size 8 --n-layer 2 \
  --csv out/pipeline.csv --json out/pipeline.json
# CUDA comparison with same initial weights and RNG seed:
python examples/101_training_pipeline_breakdown.py --device cuda
```

Each experiment samples from the same synthetic token stream, trains
an identically initialized small GPT with AdamW, and reports medians
for sample, stage/transfer, update and end-to-end latency. It also
records throughput and a **descriptive** slowest-stage label.

On CUDA, the script compares pageable/blocked, pinned/blocked, and
pinned/nonblocking transfers. The device is synchronized between
stages, so **this does not measure copy/compute overlap**. The
intermediate barriers are intentional for attribution and can distort
relative end-to-end performance versus a fully pipelined application.

This pipeline does not include corpus-file I/O, tokenizer setup, or
validation work. CPU output doesn't pretend to measure CUDA VRAM.

Run `python -m pytest tests/test_training_pipeline_breakdown.py -q`.

**Experiment:** Increase model depth and then batch size. Which stage
changes the most on your machine?

**Checkpoint:** Why can the transfer stage be a small fraction of total
time even when its standalone benchmark looks expensive?
