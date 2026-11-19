# Lesson 243 — Use a new training batch at every optimizer step

Our previous controlled optimizer experiment reused a single held-out
validation window repeatedly. That checked optimizer machinery but was
**not** a realistic training-input schedule.

Now we sample deterministic, different training windows, using a local
`torch.Generator` to avoid disturbing global random state. The same
seed recreates the same *sequence* of batches, but each optimizer step
consumes the generator's next draw.

```text
UTF-8 corpus -> character IDs -> train | validation (held out)
                                  |
                          seed-specific RNG
                                  |
                           step 1: [B,T]
                           step 2: [B,T]
                           step 3: [B,T]
```

Run `python examples/120_capstone_cli.py` to save a model, then
`python examples/136_training_parity_batches.py`.

**Question:** Why should validation batches remain excluded from
optimizer updates, even when the experiment is meant for parity?
