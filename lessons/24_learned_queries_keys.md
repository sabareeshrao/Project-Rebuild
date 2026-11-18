# Lesson 24: learn which features should be compared

The previous lesson used hand-written vectors. In a transformer,
**query** and **key** vectors are learned linear projections of each
hidden token vector.

```text
hidden [B,T,C] -> linear W_Q [C,D] -> Q [B,T,D] --+
                                                    +-> Q @ K.T -> [B,T,T]
hidden [B,T,C] -> linear W_K [C,D] -> K [B,T,D] --+
```

`C = n_embd` is the model width; `D = head_size` is the width of
one attention head. We use separate learnable matrices for Q and K.

Run `python examples/24_learned_queries_keys.py`.

**Experiment:** Change `head_size` from 3 to 2. How do Q and K change?
What happens to the *last two* score-matrix dimensions?

**Checkpoint:** Why are queries and keys projected separately, rather
than always comparing raw token embeddings?
