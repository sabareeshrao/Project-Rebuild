# Lesson 18: tokens become vector representations

In our earlier bigram model, each token selected a vector of **V next-token
logits**. Transformers instead begin with a compact, trainable vector
of length **C** (`n_embd`).

```text
token IDs [B, T] -> embedding table [V, C] -> token vectors [B, T, C]
```

The embedding table contains `V * C` trainable numbers. Repeated token
IDs select the same row. Later transformer layers will allow those
vectors to change with their **context**.

Run: `python examples/18_token_embeddings.py`

**Experiment:** Increase `n_embd` from 3 to 8. What happens to the
shape and parameter count?

**Checkpoint:** Why does a token embedding not yet understand the
surrounding words?
