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
