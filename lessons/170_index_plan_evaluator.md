# Lesson 170 — Evaluate current weights using compact start indices

An index plan is not a cache of model predictions or even a cache of
token windows. It stores the deterministic *positions* of each
validation window and regathers those windows during evaluation.

```text
CPU original corpus + cached random starts
                |
            gather [B,T] input and targets
                |
         current GPT weights → loss
```

`evaluate_index_plan_split_loss` restores training/eval mode,
disables autograd and transfers each newly gathered batch to the
model device. The equivalent two-split helper uses the same seeded
train and validation samples as the regenerated and fully cached
evaluators.

The tests compare exact loss values across **all three** approaches,
then perturb model parameters to confirm that evaluating the same
starts produces a different loss. CUDA tests skip on CPU-only runners.

The tradeoff is deliberate: tiny retained index tensors but repeated
CPU index gathering, which can be slower than cached full windows.

Run `python -m pytest tests/test_index_plan_evaluation.py -q`.

**Experiment:** Compare the three evaluators on a fixed model seed.
Why do all return the same loss?

**Checkpoint:** Why is holding validation start positions fixed
compatible with repeatedly evaluating an improving model?
