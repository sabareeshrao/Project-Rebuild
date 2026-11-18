# Lesson 34: bring many heads back to model width

We choose `n_embd = C` divisible by `n_head = H`, giving
`head_size = D = C / H`. Then H head outputs concatenate into C
features, which a learned linear projection can mix.

```text
input [B,T,C] -> H independent heads [B,T,D]
              -> concatenate [B,T,H*D] = [B,T,C]
              -> output projection [C,C] -> [B,T,C]
```

The output projection learns to combine information across heads;
it is not a softmax. The shape stays `[B,T,C]`, which will allow a
residual skip connection when we integrate this into a model.

Run: `python examples/34_multihead_projection.py`

**Experiment:** With C=12, compare H=3 and H=4. What changes about
each head's dimension, even though the final output width stays 12?

**Checkpoint:** Why must the width be divisible by the head count
in our initial GPT-style implementation?
