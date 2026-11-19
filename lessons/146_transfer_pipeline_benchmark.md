# Lesson 146 — Benchmark the whole host-to-device preparation path

A CUDA copy made from already-pinned CPU memory can look quick if the
measurement ignores *how that memory became pinned*. But a training
pipeline often has to sample tokens, allocate a batch, pin it, then
copy it to the accelerator.

Our benchmark measures the **complete batch preparation and transfer
pipeline**, using synthetic token data already in CPU RAM:

```text
CPU token stream → sample [B,T] x/y → optional pin → optional CUDA copy
              all included in wall-clock latency
```

On CUDA it compares three variants: pageable + blocking copy,
pinned + blocking copy, and pinned + nonblocking copy. It synchronizes
CUDA before each sample is recorded, including inside the operation
to keep pinned source tensors alive until the transfer completes.

On CPU it times **sampling only**, with no fabricated GPU transfer
results. Data-file I/O, model compute, and true multi-stream overlap
are not part of this experiment.

```bash
python examples/99_host_transfer_benchmark.py --device cpu
# On supported CUDA:
python examples/99_host_transfer_benchmark.py --device cuda \
  --iterations 20 --warmup 5 --csv out/transfers.csv \
  --json out/transfers.json
```

Run `python -m pytest tests/test_host_transfer_benchmark.py -q`.

**Experiment:** Compare a small and a large `--batch-size` on CUDA.

**Checkpoint:** Why shouldn't we claim asynchronous copy/compute
overlap simply because we passed `non_blocking=True`?
