# Lesson 64: crop the context, not the generated text

Position embeddings are learned for indices `0 ... block_size-1`.
If generation feeds in more positions than the model was configured
for, the position-embedding lookup will fail.

Like upstream nanoGPT, we keep all already-generated tokens in the
returned sequence but **condition on only the last block_size tokens**
at each step:

```text
entire generated sequence [B,N]
   -> last min(N, block_size) tokens
   -> GPT next-token distribution
   -> append one sampled token to entire sequence
```

This is a sliding-window context, not a KV cache. At each iteration
the model recomputes attention from the beginning of that window.

Run: `python examples/61_crop_generation_context.py`.

**Experiment:** Set `block_size=3` and generate 20 new tokens.
How long is the returned output? How many tokens are ever fed to GPT?

**Checkpoint:** Why does context cropping allow indefinite output
length without increasing the number of learned position embeddings?
