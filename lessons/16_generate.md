# Lesson 16: autoregressive generation

Generation loops one token at a time:

```text
prompt -> model scores -> softmax -> sample next token
   ^                                        |
   |____________ append token ______________|
```

A **bigram** model predicts solely from the last token ID, so earlier
context is ignored. Each new token is sampled from a categorical
distribution via `torch.multinomial`. Supplying a seeded
`torch.Generator` makes the stochastic sample sequence reproducible.

Unlike training, generation does not need gradients. The
`@torch.no_grad()` decorator avoids tracking computations.

Run: `python examples/16_generate.py`

**Experiment:** Change the random sampling seed, keeping the trained
weights fixed. Why may the output differ?

**Checkpoint:** Why must generation append the sampled token before
asking for the next token?
