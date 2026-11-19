# Lesson 303 — Adding a logit constant cannot change a prediction

Softmax cares about **differences** between vocabulary logits.
Adding the same constant to every vocabulary score for one token
position cannot alter any probability.

```text
logits:          [2, 1, -1]
shifted logits:  [9, 8,  6]  (+7 to every class)
softmax:         exactly the same distribution (up to roundoff)
```

We compare probabilities, ignored-target cross-entropy and gradients
before and after such a shift. This helps explain why stable log-softmax
subtracts a maximum before exponentiating.

Run `python examples/196_logit_shift.py`.

**Experiment:** What changes if you add different offsets to different
*classes* instead of a common offset to all classes?

**Checkpoint:** Why should an implementation use `logsumexp` or
`log_softmax` instead of multiplying many small probabilities?
