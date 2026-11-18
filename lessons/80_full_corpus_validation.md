# Lesson 80: evaluate every held-out next-token target

Randomly sampled evaluation batches are convenient, but can miss
some tokens and can count other tokens more than once. For small
corpora we can score **every adjacent token pair exactly once**.

We split a token stream into consecutive chunks of at most
`block_size` input positions. Each chunk predicts its corresponding
next tokens and contributes a **sum** of cross-entropies. We divide
the total loss by `len(tokens) - 1` after processing all chunks.

```text
tokens  [a,b,c,d,e,f,g]
inputs  [a,b,c] [d,e,f]
targets [b,c,d] [e,f,g]
```

Contexts **reset at chunk boundaries**; we do not use all possible
overlapping sliding windows, so this measure is exact for the defined
*chunked evaluation policy*, not for unlimited-context likelihood.

The model enters evaluation mode with `torch.no_grad()`, and its
original training mode is restored afterward. No random batches are
sampled, and every held-out target receives equal weight.

Run: `python examples/77_full_corpus_validation.py`.

**Experiment:** Change the chunk size from 8 to 4. Why can the
resulting loss change even though every target is still scored once?

**Checkpoint:** Why must we weight chunk losses by their token count
instead of averaging the mean loss of a short final chunk equally?
