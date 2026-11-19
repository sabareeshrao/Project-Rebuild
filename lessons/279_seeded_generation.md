# Lesson 279: repeatable sampling needs a private random generator

A language model returns probabilities; text generation samples
new tokens from those probabilities. Two runs can differ *even with
identical model weights* if the random choices differ.

```text
GPT logits -> softmax -> multinomial sampling
                           |
                 torch.Generator(seed=41)
                           |
             same prompt + seed -> same token IDs
```

We now test this rule by creating two independent generators with
the same seed. We also verify the prompt remains unchanged and the
global PyTorch CPU RNG does not get consumed.

Run `python examples/172_seeded_generation.py`.

The negative test deliberately ignores the supplied generator and
makes outputs depend on a call counter; the invariant must fail.

**Checkpoint:** Why does seeded generation prove reproducibility
of sampling, but not guarantee any particular text is high quality?
