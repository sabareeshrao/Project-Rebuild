# Lesson 09: a probability distribution from token frequencies

A distribution assigns a nonnegative probability to each token, and all
probabilities sum to 1.

With training IDs `[0, 0, 1]` and vocabulary `[0, 1, 2]`:

```text
raw counts:       [2, 1, 0]
add-one counts:   [3, 2, 1]
probabilities:    [3/6, 2/6, 1/6]
```

Adding one count to each token is **Laplace (add-one) smoothing**.
It gives an unseen token a small but nonzero probability.

This **unigram baseline** ignores context, but it uses real evidence:
frequent training tokens are assigned higher probability. Estimate it
using **training data only**, never from the held-out validation segment.

Run: `python examples/09_unigram_probabilities.py`

**Experiment:** Change the story by repeating one character. How does
its estimated probability change?

**Checkpoint:** Why is an unseen token's probability nonzero here?
