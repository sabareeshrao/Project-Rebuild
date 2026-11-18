# Lesson 32: prove that attention can use earlier context

A bigram model only sees the **current token**. Our previous
position-aware model additionally sees the **absolute position**.
Neither can distinguish the following inputs at position 1:

```text
inputs:     [0, 1]       [2, 1]
targets:    [1, 0]       [1, 2]
                 ^            ^
       same current token and same position
       but different desired next-token answers
```

The single-head model can use the first input token when computing the
second token's representation. We train it on this tiny deterministic
toy dataset for 140 optimizer steps, then verify that it predicts both
target sequences. This is a **capability test**, not a realistic
language-quality benchmark.

Run: `python examples/32_train_attention_model.py`

**Experiment:** Replace the attention output with zeros in
`SingleHeadLanguageModel.forward`, then retrain. Can a
token-plus-position-only model fit both contexts simultaneously?
Restore the original code after the experiment.

**Checkpoint:** Explain precisely which operation creates
context-dependence and how the causal mask avoids leakage from future
tokens.

**Next:** Multi-head attention, projections and normalization will
move us closer to nanoGPT's actual transformer blocks.
