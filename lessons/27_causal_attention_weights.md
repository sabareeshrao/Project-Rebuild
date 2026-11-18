# Lesson 27: combine QK, scaling, masking and softmax

This is our first **trainable attention-weight calculator**:

```text
hidden [B,T,C]
  |-- W_Q --> Q [B,T,D] ----+
  |-- W_K --> K [B,T,D] ----+--> Q @ K.T / sqrt(D)
                                      |
                           causal mask (future = -inf)
                                      |
                              softmax across keys
                                      |
                             weights [B,T,T]
```

Each row is a **probability distribution** over the key positions that
the corresponding query is allowed to see. Every row sums to one.
All future-token weights are exactly zero.

We also test **prefix independence**: changing future hidden vectors
must not change the attention weights for earlier positions.

Run `python examples/27_causal_attention_weights.py`.

**Experiment:** For a four-token input, how many nonzero key weights
can the first, second, third and fourth query rows have?

**Checkpoint:** We have query and key vectors, but **not** value
vectors or the weighted sum of values. Why does this mean we haven't
implemented a full attention head yet?

**Next:** Build `V` projections and compute `weights @ values`.
Later we will implement multiple heads and transformer blocks.
