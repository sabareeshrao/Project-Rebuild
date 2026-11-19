# Lesson 301 — What exactly does GPT sample?

A GPT produces **logits**, not a finished word. The sampling pipeline
uses three operations to turn those logits into candidate tokens:

```text
GPT last-position logits
       -> divide by temperature
       -> mask tokens below top-k threshold
       -> softmax: probabilities sum to one
```

The new inspector gives the support (allowed token IDs), peak token,
full probability vector and Shannon entropy in **nats**. It does not
actually sample a token or consume global random state.

Smaller temperature concentrates probabilities on likely tokens.
With `top_k=1`, only the highest-logit ID has nonzero probability,
giving zero entropy (except tied peaks, which our top-k filter retains).

Run `python examples/194_next_token_distribution.py`.

**Experiment:** Keep the same logits and `top_k=2`, but compare
temperatures 0.2 and 2.0. Which distribution has higher entropy?

**Caution:** Higher sampling entropy does not mean higher model quality.
