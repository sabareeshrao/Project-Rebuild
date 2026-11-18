# Lesson 31: use a causal attention head for next-token prediction

Our previous embedding language model knew the **current token** and
its **position**, but could not look at preceding tokens. This version
adds one complete causal attention head.

```text
token IDs [B,T]
 -> token + position vectors [B,T,C]
 -> causal attention [B,T,D]
 -> output projection [B,T,C]
 -> residual addition [B,T,C]
 -> vocabulary logits [B,T,V]
 -> cross-entropy against next-token targets
```

A **residual connection** adds the original hidden representation to
the attention output after projecting back into model width C. This
allows information to bypass the attention pathway.

Tests show that changing **past** tokens can change predictions at a
later position, while changing **future** tokens cannot change
earlier predictions.

Run: `python examples/31_attention_language_model.py`

**Experiment:** Compare predictions for `[0,1]` versus `[2,1]`.
Both sequences end with token 1. Why can the final-token scores differ
with attention even though they could not with our old bigram model?

**Checkpoint:** What does the residual path contribute, and what
important GPT components (multi-head attention, layer norm, MLP)
are still missing?
