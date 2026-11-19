# Lesson 236: check the model you actually trained

Earlier training-parity experiments could start from freshly initialized
weights. Here we load the **saved capstone checkpoint** and compare
a single optimizer step against a separately supplied original nanoGPT
model with synchronized parameters.

```text
model.pt (+ vocabulary) + held-out corpus
                  |
         deterministic token batch
                  |
         local GPT + original nanoGPT
                  |
       matching backward gradients
                  |
         identical plain AdamW step
                  |
        compare updated parameters
```

The experiment makes changes **in memory only**. The original
`model.pt` is not changed. The held-out batch is used solely as
a controlled comparison, not to continue fitting or evaluate
generalization.

```bash
python examples/120_capstone_cli.py
python examples/129_saved_optimizer_parity.py \
  --upstream-model /trusted/nanoGPT/model.py
```

The upstream Python file is *executed* when imported. Review it first.

**Checkpoint:** Why is the original checkpoint unchanged even though
the training-parity function performs an AdamW optimizer step?
