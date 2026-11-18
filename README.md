# Project Rebuild: nanoGPT

An original, learning-first reimplementation of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). This is **not** a fork or a copy of its Git history.

## Goal

Understand how a GPT language model works by implementing each concept ourselves, validating it with small examples and tests, and studying the original project at checkpoints.

## Our commit rules

1. One clear learning objective per commit.
2. Explain **why** the change exists, not just how to type it.
3. Favor small, runnable steps; add tests as soon as behavior appears.
4. Use original code and diagrams where helpful, with proper attribution.
5. Compare our eventual implementation against nanoGPT's behavior.

## First learning milestones

- Set up a reproducible Python and PyTorch project.
- Represent token IDs as tensors shaped **[batch, time]**.
- Write tests for shapes and dtypes.
- Construct next-token prediction inputs and targets.
- Gradually implement embeddings, attention, transformers, training and generation.

## Attribution

This independent educational project is inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). We will not pretend our commits are the upstream project's historical commits.

## Local setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev]"
```

PyTorch is an explicit project dependency; a GPU is **not** required for the first lessons.

## First runnable lessons

```bash
python examples/01_token_batches.py
python examples/02_next_token_pairs.py
python -m pytest
```

Each lesson in `lessons/` explains the problem, a code experiment, and a
question to answer before moving to the next commit.

## Text-to-batch pipeline (commits 006–010)

Run the new learning exercises from the repository root after installation:

```bash
python examples/03_character_vocabulary.py
python examples/04_encode_decode.py
python examples/05_load_corpus.py
python examples/06_split_train_val.py
python examples/07_sample_batch.py
python -m pytest
```

Read `lessons/03_character_vocabulary.md` through `lessons/07_sample_batch.md`
in order. The checked-in `data/tiny_story.txt` is original example text.
For now we use **character-level** tokenization to learn the fundamentals;
later phases will cover nanoGPT-compatible tokenization and transformers.

## Prediction and evaluation (commits 011–015)

Before building a learnable transformer, we establish a measurable baseline:

```text
tokens -> context windows [B,T]
       -> next-token scores (logits) [B,T,V]
       -> probabilities (softmax over V)
       -> compare with targets [B,T] using cross-entropy
```

Run the new examples from the repository root:

```bash
python examples/08_uniform_baseline.py
python examples/09_unigram_probabilities.py
python examples/10_softmax.py
python examples/11_cross_entropy.py
python examples/12_compare_baselines.py
python -m pytest
```

The toy baselines have **no learned neural-network weights**. We use
training data only to estimate unigram frequencies, then compare both
baselines on the same held-out batch. Read lessons 08–12 in `lessons/`
for equations, experiments, and understanding checkpoints.

## First trainable language model (commits 016–020)

The `BigramLanguageModel` learns a `[V,V]` next-token lookup table.
These lessons introduce the first **learnable weights**, gradient updates
with AdamW, autoregressive token sampling, and an end-to-end training
exercise. This is a pedagogical bigram model, **not yet a transformer**.

```bash
python examples/13_bigram_lookup.py
python examples/14_bigram_loss.py
python examples/15_train_step.py
python examples/16_generate.py
python examples/17_train_bigram.py
python -m pytest
```

Read `lessons/13_bigram_lookup.md` through
`lessons/17_train_bigram.md` before proceeding to vector embeddings.

## Token vectors and learned positions (commits 021–025)

These five lessons extend token IDs into trainable embedding vectors,
add a projection head, and incorporate absolute positions:

```text
token IDs [B,T] -> token vectors [B,T,C] --------+
positions [T] -> position vectors [T,C] ----------+-> sum [B,T,C]
                                                        |
                                                linear output head
                                                        |
                                                logits [B,T,V]
```

They culminate in a small position-dependent training experiment:

```bash
python examples/18_token_embeddings.py
python examples/19_output_projection.py
python examples/20_position_embeddings.py
python examples/21_position_aware_model.py
python examples/22_train_embedding_model.py
python -m pytest
```

Read `lessons/18_token_embeddings.md` through
`lessons/22_train_embedding_model.md`. **Important:** this model cannot
yet condition on *previous* tokens. Causal self-attention is the next
architectural milestone.

## Query-key mathematics (commits 026–030)

We now have a trainable **causal attention-weight calculator**.
These lessons introduce dot products, separate query and key
projections, inverse-square-root scaling, a future-token mask,
and row-wise softmax.

```text
hidden [B,T,C] -> Q,K [B,T,D] -> Q @ K.T [B,T,T]
               -> scale -> causal mask -> softmax -> weights [B,T,T]
```

Run:

```bash
python examples/23_pairwise_dot_scores.py
python examples/24_learned_queries_keys.py
python examples/25_scaled_dot_scores.py
python examples/26_causal_mask.py
python examples/27_causal_attention_weights.py
python -m pytest
```

Read `lessons/23_pairwise_dot_scores.md` through
`lessons/27_causal_attention_weights.md`.

**Not a full attention head yet:** we have not projected value vectors
or computed the weighted aggregation `weights @ values`. Those are
the next lessons on our path toward nanoGPT.

## Complete single-head causal self-attention (commits 031–035)

We now produce **value vectors**, apply learned **causal attention
weights** to them, and integrate the resulting single-head
representation into a next-token language model with a residual path.

```text
hidden [B,T,C] -> Q,K [B,T,D] -> scaled causal weights [B,T,T] --+
hidden [B,T,C] -> V [B,T,D] -------------------------------+------+
                                                          |
                                                   weights @ V [B,T,D]
                                                          |
                                                  linear projection [B,T,C]
                                                          |
                                             residual + hidden [B,T,C]
                                                          |
                                                   logits [B,T,V]
```

Run the new examples (after installing the project):

```bash
python examples/28_value_projection.py
python examples/29_weighted_values.py
python examples/30_single_attention_head.py
python examples/31_attention_language_model.py
python examples/32_train_attention_model.py
python -m pytest
```

Read `lessons/28_value_projection.md` through
`lessons/32_train_attention_model.md`. One experiment deliberately
requires earlier-token context to disambiguate next-token targets.

**Scope:** This is one complete **causal attention head**, not a full
GPT-2 transformer. We still need multiple heads, layer normalization,
feed-forward networks, stacked blocks, training refinements, and
compatibility checks with original nanoGPT.

## Multi-head attention (commits 036–040)

We now concatenate **multiple independent causal self-attention heads**,
learn a projection back into the hidden width, optionally apply
training-only **output dropout**, and combine the result with token +
position embeddings through a residual connection.

```text
hidden [B,T,C] -> H heads [B,T,C/H] -> concat [B,T,C]
              -> projection [B,T,C] -> dropout -> residual
              -> vocabulary logits [B,T,V] -> next-token loss
```

New lessons and runnable exercises:

```bash
python examples/33_parallel_heads.py
python examples/34_multihead_projection.py
python examples/35_attention_dropout.py
python examples/36_multihead_language_model.py
python examples/37_train_multihead_model.py
python -m pytest
```

Read `lessons/33_parallel_heads.md` through
`lessons/37_train_multihead_model.md`. The final exercise trains on
two sequences with identical last input tokens but different required
next-token predictions, confirming the model can use earlier context.

**Not yet a GPT transformer block:** LayerNorm, a feed-forward MLP,
stacked blocks and efficient fused attention remain to be built.
