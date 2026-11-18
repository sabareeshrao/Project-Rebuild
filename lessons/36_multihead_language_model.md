# Lesson 36: a residual language model with multiple heads

We replace the earlier single attention head with **multi-head
causal attention** while keeping the learned token embeddings,
absolute position embeddings, residual skip, and vocabulary head.

```text
tokens [B,T] -> token + position vectors [B,T,C]
                                        |
                            multi-head causal attention [B,T,C]
                                        |
                              add residual [B,T,C]
                                        |
                              output head [B,T,V]
                                        |
                         next-token cross-entropy
```

Each attention head can gather past-token information differently.
The output projection recombines their features before the residual
addition. The same `train_step` helper from Lesson 15 still works.

Run: `python examples/36_multihead_language_model.py`

**Experiment:** Change `n_head` from 2 to 4 while holding
`n_embd=8`. What happens to each head's width and to the logits shape?

**Checkpoint:** Why are attention outputs causal across the whole
model, despite stacking multiple independently learned heads?

**Missing GPT pieces:** LayerNorm, a feed-forward network, deeper
transformer blocks, and more efficient combined QKV projections.
