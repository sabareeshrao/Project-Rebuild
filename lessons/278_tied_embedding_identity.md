# Lesson 278: one parameter performs two jobs

In nanoGPT, the token lookup table and output vocabulary projection
share the **same PyTorch Parameter**.

```text
token IDs -> embedding.weight --+--> Transformer blocks
                                |
                                +--> output head (same weight)
```

Two tensors that merely start with equal numbers are not *tied*:
after an optimizer step, their independent gradients can separate.

Our invariant checks object identity and confirms the shared weight
appears once among the trainable parameters.

Run `python examples/171_embedding_tying.py`.

**Experiment:** Clone the output head's weight into a new
`nn.Parameter` without changing its values. Why does the identity
check fail even though the matrices look identical?
