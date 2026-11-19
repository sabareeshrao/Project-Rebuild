# Lesson 239: compare REAL optimizer groups

A matching rule on paper is not enough: inspect the parameters
*actually* passed to two instantiated optimizers.

Our local `configure_adamw` and a separately provided nanoGPT
`configure_optimizers` both return AdamW. We translate the local
names into nanoGPT keys and compare the weight-decay value assigned
to each unique trainable tensor, independent of group ordering.

```text
our GPT -> our AdamW groups --names--> [parameter -> decay]
original -> original groups ---------> [parameter -> decay]
                                         |
                                    compare keys
```

Run with **reviewed** upstream Python:
```bash
python examples/132_optimizer_group_report.py \
  --upstream-model /trusted/nanoGPT/model.py
```

This executes the trusted source; it does not fetch anything
automatically. Local CI tests use a naming adapter, not the actual
upstream implementation.

**Checkpoint:** Why must we inspect group membership rather than
simply count the number of decay and no-decay tensors?
