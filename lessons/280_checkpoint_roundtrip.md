# Lesson 280: saving and reloading must preserve model behavior

An inference checkpoint stores **weights, GPT configuration, and
tokenizer vocabulary**. Reading and saving the same checkpoint should
not alter any of them.

```text
saved.pt -> load -> model + tokenizer
                  |
             save temporary.pt
                  |
                  +-> reload -> compare EVERY state tensor
                              -> compare vocab/config
                              -> compare predicted logits
```

We create the second checkpoint in a temporary directory, and never
modify the original. Its serialized file bytes need not match
byte-for-byte; the *meaningful deserialized state* must match.

The check also preserves global CPU RNG state, since initializing
models during load otherwise consumes random numbers.

Run `python examples/173_checkpoint_roundtrip.py` after creating
`out/capstone/model.pt`.

**Checkpoint:** Why does comparing tensor values and forward logits
provide stronger evidence than comparing raw .pt file hashes?
