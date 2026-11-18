# Lesson 30: our first complete causal self-attention head

We can now connect every part of a single **causal attention head**:

```text
hidden [B,T,C] -> Q and K [B,T,D] -> QKᵀ / √D
                                     -> causal mask
                                     -> softmax -> weights [B,T,T]
hidden [B,T,C] -> V [B,T,D] -----------+
                                      |
                       weights @ V -> output [B,T,D]
```

The Q and K projections control **where** each token gathers
information. The V projection determines **what information** it
gathers. Earlier tokens are available; future tokens are blocked.

The code is checked against a direct PyTorch calculation, and a
**prefix-invariance test** changes future inputs while asserting
that earlier outputs stay unchanged.

Run: `python examples/30_single_attention_head.py`

**Experiment:** Check the first row of attention weights. Why can
position 0 only attend to itself?

**Checkpoint:** Why is a complete single head *still not a full
transformer block*? (Think multi-head attention, residuals, norm,
and feed-forward transformations.)
