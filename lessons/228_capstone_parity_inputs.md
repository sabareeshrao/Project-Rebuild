# Lesson 228: use the SAME input for both GPTs

A meaningful implementation comparison must use the same weights,
input token IDs, and targets. We draw deterministic **validation**
windows from the actual capstone corpus and check that the saved
tokenizer's vocabulary still matches that corpus.

```text
held-out text -> token stream -> seeded windows [B, T]
                                     | -> input IDs [B, T]
                                     + -> target IDs [B, T]
```

Set the final target at each row to `-1` to exercise upstream
nanoGPT's ignored-target convention. We still feed every input
position through the model; only the final loss target is masked.

First generate a capstone checkpoint with
`python examples/120_capstone_cli.py`, then run
`python examples/121_capstone_parity_inputs.py`.

**Question:** Why does matching random seeds alone not establish parity
if the models are given different token IDs?
