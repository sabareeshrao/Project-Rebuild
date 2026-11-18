# Lesson 45: build a configurable multi-layer GPT

The `GPTLanguageModel` uses the same building blocks we've already
tested, but now stacks as many transformer blocks as its config asks
for. It does not change the original one-block model, which remains
available for comparisons.

```text
token IDs [B,T]
 -> token + position embeddings [B,T,C]
 -> input dropout
 -> n_layer causal transformer blocks [B,T,C]
 -> final LayerNorm [B,T,C]
 -> vocabulary logits [B,T,V]
 -> next-token loss
```

Each block has separately learned Q/K/V attention projections,
a causal mask, a GELU feed-forward network, and residual connections.
This is a readable decoder-only transformer, **not yet an exact
line-by-line port** of nanoGPT.

Run: `python examples/45_stacked_gpt.py`.

**Experiment:** Change `n_layer=3` to 1 and compare logits shapes
and the number of block modules. Does the number of model parameters
stay the same?

**Checkpoint:** Why can adding layers increase what a model computes
without increasing the feature width at each token?
