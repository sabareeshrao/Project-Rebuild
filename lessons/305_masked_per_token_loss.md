# Lesson 305 — Read next-token loss one token at a time

A single mean GPT loss can hide where the model is confident and
where it is wrong. Cross-entropy is a **mean negative log-likelihood
(NLL)** over supervised positions, not a mean of row averages.

```text
targets:            [a, -1, -1]     [b, c, -1]
supervised counts:       1               2
mean cross-entropy = sum(three nonignored NLLs) / 3
```

Our diagnostic returns each token NLL, marks ignored targets as
`None`, and reports per-sequence means and valid-token counts.
An entirely ignored sequence has an undefined per-sequence loss,
not an artificial zero.

Run `python examples/198_token_loss_breakdown.py`.

**Question:** Why is averaging two sequence means misleading when
one sequence has one supervised position and the other has ten?
