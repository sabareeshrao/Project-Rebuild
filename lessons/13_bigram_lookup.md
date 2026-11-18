# Lesson 13: your first learnable language model

Our uniform and unigram baselines returned the **same** distribution
for every context position. A bigram model changes that: the **current
token** selects a row of learnable next-token scores.

```text
              next token: a  b  c  d
current a             [ ... ... ... ... ]
current b             [ ... ... ... ... ]
current c             [ ... ... ... ... ]
current d             [ ... ... ... ... ]
```

The embedding table is a matrix of shape `[V, V]`, not yet the
lower-dimensional vector embeddings used in transformers. Token IDs
`[B, T]` select rows and yield logits `[B, T, V]`. The table has
`V * V` trainable values.

Run: `python examples/13_bigram_lookup.py`

**Experiment:** Change the first token while keeping the second token
identical. Which prediction scores are guaranteed to stay identical?

**Checkpoint:** Why can a bigram model learn token-to-token transitions
but not depend on anything that happened two tokens ago?
