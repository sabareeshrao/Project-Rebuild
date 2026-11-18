# Lesson 65: adjust the randomness with temperature

Raw logits are not probabilities. Before softmax, we can divide them
by a positive **temperature**:

```text
p(next_token=i) = softmax(logits / temperature)[i]
```

With `temperature=1`, the original distribution is unchanged.
Lower positive temperatures sharpen the distribution and emphasize
the largest logit; higher temperatures flatten it.

This changes the sampling probabilities, not the model weights.
We deliberately reject zero, negative, NaN and infinite
temperatures. A separate greedy mode can be added later.

Run: `python examples/62_sampling_temperature.py`.

**Experiment:** Compare probabilities for logits `[0,1,3]`
at temperature 0.5 and 2.0.

**Checkpoint:** Why do we apply temperature **before** softmax,
rather than dividing probabilities afterward?
