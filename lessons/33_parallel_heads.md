# Lesson 33: several attention heads in parallel

One causal attention head mixes information from earlier tokens. We
can let **H independently parameterized heads** do this in parallel,
then concatenate their results along the feature axis.

```text
hidden [B,T,C] -> head 1 [B,T,D] --+
               -> head 2 [B,T,D] --+--> concat [B,T,H*D]
               -> head H [B,T,D] --+
```

Different heads have distinct Q, K, and V parameters. They *can*
learn different relationships; merely creating multiple heads does
not guarantee they specialize.

Run: `python examples/33_parallel_heads.py`

**Experiment:** Change H from 3 to 4. Which output dimension changes?
How many independent Q/K/V projections do we now have?

**Checkpoint:** Why is concatenation along the final feature axis
different from averaging the heads?

**What is missing:** The concatenated output still needs a learned
projection back to the model width, so it can enter a residual path.
