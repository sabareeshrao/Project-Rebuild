# Lesson 302 — Four GPT semantics experiments on one saved model

A GPT can make predictions with plausible shapes even when it
handles dropout, masking or weight sharing incorrectly.

Our new lab runs four independent tests against a trusted saved GPT:

```text
checkpoint + tiny held-out batch
   |-> train/eval dropout: stochastic vs repeatable
   |-> masked targets: match independently selected CE gradients
   |-> tied embeddings: two gradient paths add into shared weight
   +-> top-k and temperature: normalized sampling probabilities
                          |
               JSON evidence if all four succeed
```

For the dropout experiment, a **disposable copy** of the trained
weights uses a nonzero dropout probability, because checkpoints
trained without dropout cannot demonstrate stochastic masks.

Run `python -m rebuild_gpt.release_review_cli --mode train` into
a new directory, then `python examples/195_model_semantics.py`.

**Question:** Why do individual small-model semantic checks provide
valuable debugging evidence without establishing GPT-2 benchmark
quality or full parity with original nanoGPT?
