# Lesson 12: evaluate predictions on held-out text

Now join the concepts learned in commits 001–014:

```text
text -> token IDs -> train / validation split
     -> sample input + target windows
     -> logits [B,T,V] -> cross-entropy loss
```

The **uniform** baseline gives every next token the same score, so its
cross-entropy is `log(V)` for vocabulary size `V`.

The **unigram** baseline estimates token frequencies from **training**
text, with add-one smoothing, then uses the same probabilities at every
context position. It is more informed than uniform, but cannot model
word order or context.

Both are scored on the **same** validation batch. Lower loss means
higher average probability assigned to the actual next tokens.

Run: `python examples/12_compare_baselines.py`

**Experiment:** Change the random seed or the context length. How
much does validation loss vary on this tiny text corpus?

**Checkpoint:** Could a unigram model do worse than uniform on a held-out
dataset? Explain why the answer is yes.

**Looking ahead:** A learnable bigram model can assign different
distributions depending on the current token. That is our next
step toward context-sensitive predictions.

**Caution:** This exercise uses a toy corpus and a character vocabulary
created from the full text. It is not a rigorous real-world benchmark.
