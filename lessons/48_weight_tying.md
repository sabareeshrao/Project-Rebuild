# Lesson 48: tie input and output token weights

Our early GPT had **two** distinct matrices: input token embeddings
`[V,C]` and an output projection `[V,C]`. GPT-2-style models
often share these weights, so learning a token representation also
changes its contribution to output logits.

```text
token IDs -> embedding table W [V,C] -> hidden features
                                         |
                             logits = hidden @ W.T
```

In PyTorch, assigning `head.linear.weight = tokens.lookup.weight`
makes both modules reference **the same Parameter object**, not merely
two tensors with equal values. Its gradients and optimizer state are
shared. The model saves `V*C` trainable parameters.

For fidelity, `GPTConfig.tie_embeddings` now defaults to `True`.
Use `False` to reproduce the untied implementation from Lesson 46.
The parameter formula for the default educational attention path
is now `V*C + T*C + 2*C + L*(12*C*C + 10*C)`.

Run: `python examples/48_weight_tying.py`.

**Experiment:** Compare tied and untied parameter counts with
`vocab_size=100`; what does changing `n_embd` do to savings?

**Checkpoint:** How is sharing the same Parameter different from
copying one set of weight values to another module?

**Scope:** This moves closer to nanoGPT, but does not yet make our
model checkpoint-compatible with upstream.
