# Lesson 66: restrict sampling to the top-k candidates

A vocabulary can contain thousands of token IDs. Sometimes we
want to sample only from the tokens with the largest scores.

For each batch row, find the kth-largest **temperature-scaled logit**,
then replace every smaller logit with negative infinity:

```text
scores [1,5,0,3], k=2 -> [-inf, 5, -inf, 3]
                                     |
                                   softmax
```

Candidates below the threshold have exactly zero probability. Tokens
tied at the threshold remain eligible, matching upstream nanoGPT's
behavior; therefore ties can leave more than k possible choices.

`top_k=None` disables filtering, and `top_k > vocab_size`
does not remove any candidate. A value of 1 usually selects
the maximum-scoring token (greedy decoding without score ties).

Run: `python examples/63_sampling_topk.py`.

**Experiment:** Use the same model and prompt, then compare top-k=1,
top-k=3 and no top-k across different random seeds.

**Checkpoint:** Why apply top-k **before** softmax rather than
randomly discarding probabilities afterward?
