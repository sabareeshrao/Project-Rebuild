# Lesson 42: assemble and train our first GPT-style block

Our components now form an end-to-end **one-block** decoder-only
language model:

```text
token IDs [B,T]
 -> token + absolute position embeddings [B,T,C]
 -> input dropout
 -> pre-LayerNorm causal transformer block [B,T,C]
      -> masked multi-head self-attention + residual
      -> feed-forward GELU MLP + residual
 -> final learned LayerNorm [B,T,C]
 -> vocabulary projection [B,T,V]
 -> next-token cross-entropy -> AdamW updates
```

We reuse the same tiny context-sensitive task from earlier lessons:

```text
inputs [0,1] -> targets [1,0]
inputs [2,1] -> targets [1,2]
```

At position 1, both inputs contain the same current token and position.
The correct target differs because the **earlier token differs**.
Causal attention must supply that distinguishing context. We test that
the training loss falls and the model predicts both target sequences.

Run: `python examples/42_train_one_block.py`.

**Experiment:** Set the attention output projection weights to zero
before training. Can the remaining token/position+MLP path solve both
contexts at their second positions? Restore attention afterward.

**Checkpoint:** Identify exactly where time steps can exchange
information, and where operations remain position-wise.

**Next:** Stack multiple transformer blocks, add configurable model
depth and count parameters. Later, study fused QKV and efficient
attention implementations to approach original nanoGPT.
