# Lesson 172 — Compare three validation storage strategies

Character-GPT now has three explicit sampled-evaluation backends:

- `regenerate` (default): redraw the same seeded start vectors and
  gather token batches at each evaluation event.
- `cached`: keep the sampled input/target tensors on CPU.
- `indices`: keep only sampled start vectors on CPU and reconstruct
  windows on demand.

Run:

```bash
python -m rebuild_gpt.train_cli --steps 5 --eval-every 1 \
  --eval-batch-backend indices --eval-cache-max-mib 1

python examples/106_index_eval_benchmark.py \
  --eval-batches 4 --batch-size 8 --context 32 \
  --csv out/eval_backends.csv --json out/eval_backends.json
```

The index backend may be especially attractive for large context
lengths and tight host-memory budgets. It retains `B × K × 8`
bytes per split rather than `2 × B × T × K × 8` logical cached
token bytes. Both reuse the corpus tensor already loaded for training.

The benchmark verifies identical validation losses, measures
evaluation-only median latency, and reports retained plan bytes. It
does not include cache preparation, corpus loading, or optimizer
updates. Fewer retained bytes are **not a guarantee of faster
training**.

The CLI records the selected backend, its appropriate estimated
retained bytes and any optional cap; resume checks reject mismatches.

Run `python -m pytest tests/test_index_eval_cli.py \
tests/test_index_eval_benchmark.py -q`.

**Experiment:** Increase context from 4 to 64. Compare retained bytes
and evaluation latency for all three choices.

**Checkpoint:** Why might the `indices` backend use less memory but
do more work during each evaluation than `cached`?
