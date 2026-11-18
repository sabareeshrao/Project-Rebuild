# Lesson 01: the shape of language-model input

**Question:** How do we present multiple token sequences to a GPT model?

Tokenization turns text into integer IDs. For now we use made-up IDs so the
tensor structure is visible without having to implement a tokenizer.

```text
sequence A: 10  20  30  40
sequence B: 50  60  70  80
            └── time ───┘
            2 rows × 4 positions
```

A batch has shape **(B, T)**: B sequences and T token positions. We store
token IDs as `torch.long` (64-bit integers) because embedding-table lookups
use integer indices.

Run `python examples/01_token_batches.py` after installing the package.

**Experiment:** Add a third sequence. Which dimension changes?

**Checkpoint:** Explain the difference between batch size and context length.
