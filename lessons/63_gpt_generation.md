# Lesson 63: GPT writes one token at a time

A GPT predicts a distribution over the **next** token. Generation
repeats three operations:

```text
prompt [B,T]
  -> final-position logits [B,V]
  -> softmax + categorical sample [B,1]
  -> append to prompt [B,T+1] -> repeat
```

Unlike the older bigram generator, GPT conditions its scores on
the entire available prefix via causal attention. Sampling is random,
so a seeded `torch.Generator` helps us reproduce an experiment.
The `@torch.no_grad()` decorator avoids building a gradient graph.

The model should be put in `eval()` mode for inference; otherwise
dropout changes the distributions between samples.

Run: `python examples/60_generate_gpt.py`.

**Experiment:** Change the generator seed and compare the new tokens.

**Checkpoint:** Why do we feed the generated token back into the
model instead of sampling all future tokens from one prediction?

**Current limit:** Inputs cannot exceed the configured `block_size`.
The next lesson adds sliding-context cropping.
