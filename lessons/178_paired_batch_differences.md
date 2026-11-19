# Lesson 178 — Inspect *each* paired validation batch

Comparing just the mean loss across validation seeds can conceal a
lot. A new model might improve on some sampled windows and worsen
on others while producing a small overall difference.

We now support `compare_paired_batches`: each seed produces a fixed
index plan, and both models evaluate **the same batches, in order**.

```text
seed → shared validation batch 1 → loss A₁ vs B₁ → delta₁
     → shared validation batch 2 → loss A₂ vs B₂ → delta₂
     → shared validation batch 3 → loss A₃ vs B₃ → delta₃
```

A negative candidate-minus-baseline loss means the candidate did
better on that batch. Per-seed averages retain the original sampled
comparison's arithmetic and match its reported deltas exactly.

Results include per-seed baseline and candidate batch losses, per-batch
deltas, per-seed mean differences, and descriptive variation across
seeds. The underlying code leaves model weights, training modes and
global RNG untouched.

Run `python -m pytest tests/test_paired_batch_evaluation.py -q`.

**Experiment:** Perturb model weights and inspect whether every
validation batch moves in the same direction.

**Checkpoint:** Why should different batches be evaluated in the
same order by both models?
