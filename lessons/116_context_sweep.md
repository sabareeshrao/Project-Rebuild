# Lesson 116 — How attention latency changes with context length

A single benchmark shape cannot answer how attention scales. With
self-attention, the explicit score matrix has shape `[B,H,T,T]`,
so its *score storage* grows quadratically with context length T.

We now sweep several T values and benchmark complete GPT forward
passes with identical weights for manual and SDPA implementations.
The script reports median milliseconds, token throughput and the
maximum logit difference, and can save machine-readable CSV.

```bash
python examples/91_context_sweep.py --lengths 4,8,16 --iterations 5
python examples/91_context_sweep.py --lengths 4,8,16 --csv out/sweep.csv
```

Use the CSV to create your own plot of context length versus latency.
No speedup is assumed: PyTorch can select different SDPA kernels on
different hardware, and small CPU models may be dominated by other
layers and Python overhead.

**Experiment:** Compare 4, 8 and 16 positions and inspect latency
per input token.

**Checkpoint:** Why doesn't doubling T necessarily double or
quadruple the *entire GPT forward latency*, even though the explicit
attention score matrix is quadratic in T?
