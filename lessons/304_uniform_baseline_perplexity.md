# Lesson 304 — Perplexity gives cross-entropy an intuitive scale

If a model chooses uniformly among `V` next-token choices, every
correct token has probability `1/V`.

```text
cross entropy = -mean(log P(target)) = ln(V)
perplexity    = exp(cross entropy)    = V
bits/token    = cross entropy / ln(2)
```

Perplexity is an exponentiated loss, **not** a classification
accuracy or a benchmark score by itself. Smaller values indicate
higher assigned likelihood on the tested target tokens.

The new helper compares measured loss to uniform guessing using
only the supervised positions, ignoring target ID `-1`.

Run `python examples/197_uniform_perplexity.py`.

**Exercise:** Compare a model confident in the correct token with
one equally confident in the wrong token. Which can have perplexity
greater than the uniform baseline?

**Checkpoint:** Why must two character-level models use comparable
vocabularies and evaluation distributions before their perplexities
can be fairly compared?
