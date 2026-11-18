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

## First GPT-style transformer block (commits 041–045)

We now have a complete **pre-LayerNorm causal transformer block**,
with one multi-head attention sublayer and one GELU feed-forward MLP.
Both have residual connections. A one-block language model applies
token/position embeddings, the block, final LayerNorm, vocabulary
projection and cross-entropy.

```text
token IDs [B,T] -> embeddings [B,T,C]
  -> LayerNorm -> multi-head attention -> residual add
  -> LayerNorm -> GELU feed-forward   -> residual add
  -> final LayerNorm -> logits [B,T,V] -> cross-entropy
```

Run:

```bash
python examples/38_feature_normalization.py
python examples/39_affine_layernorm.py
python examples/40_feedforward.py
python examples/41_transformer_block.py
python examples/42_train_one_block.py
python -m pytest
```

Read `lessons/38_feature_normalization.md` through
`lessons/42_train_one_block.md`. The toy training experiment checks
whether attention can use an earlier token to predict the next one.

**Not yet original nanoGPT:** We use a single readable block and
separate attention heads. Stacking blocks, efficiency improvements,
training infrastructure, and checkpoint compatibility come later.

## Stacked GPT architecture (commits 046–050)

We now have a configurable **multi-block decoder-only GPT** built
from our previous lessons, without changing the earlier one-block
model. Each transformer block has its own learnable parameters.

```text
tokens [B,T] -> token/position embeddings [B,T,C]
             -> n_layer transformer blocks [B,T,C]
             -> final LayerNorm -> logits [B,T,V]
             -> next-token cross-entropy + AdamW
```

Run the five new lessons:

```bash
python examples/43_transformer_stack.py
python examples/44_gpt_config.py
python examples/45_stacked_gpt.py
python examples/46_count_parameters.py
python examples/47_train_stacked_gpt.py
python -m pytest
```

Read `lessons/43_transformer_stack.md` through
`lessons/47_train_stacked_gpt.md`. Tests cover separate block
parameters, configuration validation, parameter counts, gradient
flow, **prefix causality through several blocks**, and a tiny
context-dependent training task.

**Compatibility note:** Our readable architecture currently uses
untied output embeddings, and separate Q/K/V projections.
These are intentional learning simplifications; the final model
does not yet match nanoGPT checkpoint parameter names and layouts.

## Toward nanoGPT fidelity (commits 051–055)

These lessons add **tied token/output embedding weights**, GPT-style
normal weight initialization (with depth-scaled residual projections),
packed QKV projection, batched causal multi-head attention, and an
optional attention implementation switch.

```text
GPTConfig(..., tie_embeddings=True, attention_impl="educational")
GPTConfig(..., tie_embeddings=True, attention_impl="fused")
```

The default remains the original readable attention implementation,
so the separate-head learning exercises still work. Choose `"fused"`
to use a single `c_attn` per block and batched attention scores.
The fused variant includes QKV biases, so parameter counts differ
slightly. Neither mode is guaranteed to load upstream nanoGPT weights
yet: module names, state-dict structure and config details differ.

```bash
python examples/48_weight_tying.py
python examples/49_gpt_initialization.py
python examples/50_fused_qkv.py
python examples/51_fused_causal_attention.py
python examples/52_compare_gpt_attention.py
python -m pytest
```

Read `lessons/48_weight_tying.md` through
`lessons/52_compare_gpt_attention.md` for the exact equations,
weight-layout differences, reproducible experiments, and tests.

## Checkpoints and state-dict compatibility (commits 056–060)

You can now inspect named tensors and shared parameters, save and
strictly reload our **own** GPT checkpoints, resume single-process
CPU AdamW training (including saved RNG states), and translate the
fused model's tensor keys toward the **original nanoGPT** layout.

```text
Our fused GPT -> state_dict -> translate keys -> nanoGPT-style tensors
                                      |
                              import back to our GPT
                                      |
                              check identical logits
```

Run the five lessons:

```bash
python examples/53_inspect_state.py
python examples/54_save_load_model.py
python examples/55_resume_training.py
python examples/56_nanogpt_keys.py
python examples/57_checkpoint_parity.py
python -m pytest
```

Read `lessons/53_inspect_state.md` through
`lessons/57_checkpoint_parity.md`. Model checkpoints load on CPU
using `weights_only=True`. Training checkpoints currently support
standard AdamW and CPU RNG state.

**Important limitation:** Tests verify self-consistent translation
and faithful reloads of our own weights. We have **not yet loaded a
real original nanoGPT checkpoint**, verified upstream optimizer
compatibility, or implemented all upstream configuration variants.
Use trusted checkpoint sources only.

## Compare against original nanoGPT (commits 061–065)

We added bias-free LayerNorm/MLP building blocks, a fused-GPT
`bias` option, and `forward_nanogpt()` to reproduce the original
inference/training logit and loss contract without changing the
learning-first `forward()`.

A strict parity helper can synchronize identical fused-GPT tensors
into the original nanoGPT model and compare full logits, final-token
logits, and ignored-target cross-entropy numerically. No upstream
source is vendored into this repository.

For a reproducible **opt-in** comparison, obtain and review the
original `model.py` at upstream commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`:

```bash
curl -L https://raw.githubusercontent.com/karpathy/nanoGPT/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py -o /tmp/original_nanogpt_model.py
NANOGPT_MODEL_PY=/tmp/original_nanogpt_model.py python -m pytest tests/test_actual_nanogpt_parity.py -v
python examples/59_compare_original_nanogpt.py --upstream-model /tmp/original_nanogpt_model.py
python -m pytest
```

The optional integration tests **skip** when the environment variable
is absent. Read `lessons/58_optional_bias.md` through
`lessons/62_actual_nanogpt_parity.md` to understand the work.

**Scope:** Reference tests use tiny CPU models with transferred
synthetic weights, not downloaded GPT-2 checkpoint weights. Passing
these tests does not establish full checkpoint compatibility.

## Autoregressive generation and sampling (commits 066–070)

The stacked `GPTLanguageModel` can now generate token IDs using
the same basic inference sequence as nanoGPT: take last-position
logits, apply temperature, optionally filter top-k candidates,
softmax, sample one ID, and append it to the growing output. When
the context exceeds `block_size`, only the newest `block_size`
tokens are passed through the model; all generated tokens remain
in the returned tensor.

```python
model.eval()
output_ids = model.generate(
    prompt_ids, max_new_tokens=30, temperature=0.8, top_k=10,
)
```

Run the new exercises:

```bash
python examples/60_generate_gpt.py
python examples/61_crop_generation_context.py
python examples/62_sampling_temperature.py
python examples/63_sampling_topk.py
python -m pytest
```

Read `lessons/63_gpt_generation.md` through
`lessons/67_upstream_generation_parity.md`. An **opt-in**
reference test compares sampled tokens against an independently
obtained, trusted original nanoGPT `model.py`:

```bash
NANOGPT_MODEL_PY=original_nanogpt_model.py python -m pytest tests/test_actual_nanogpt_generation.py -v
python examples/64_compare_upstream_sampling.py --upstream-model original_nanogpt_model.py
```

The reference tests skip when `NANOGPT_MODEL_PY` is absent; normal
tests are fully offline. This does not imply real GPT-2 checkpoint
compatibility. Current generation uses full-window recomputation,
not KV caching or accelerated attention.

## Text-to-text GPT inference (commits 071–075)

A trained character-level GPT can now accept a **text prompt**, generate
token IDs with temperature/top-k sampling, and decode them back into
readable characters. We save the **exact ordered tokenizer vocabulary**
alongside model weights and config in a single inference checkpoint,
so token IDs remain meaningful after reloading.

Train a small model on the included text, then generate text from a
fresh process:

```bash
python examples/68_train_text_gpt.py --steps 60 --output tiny_story_gpt.pt
python -m rebuild_gpt.text_cli \
  --checkpoint tiny_story_gpt.pt --prompt "a" \
  --new-tokens 80 --temperature 0.8 --top-k 5 --seed 13
```

Or run the full demonstration without saving a persistent checkpoint:

```bash
python examples/65_save_vocabulary.py
python examples/66_generate_text.py
python examples/67_text_checkpoint.py
python examples/69_text_to_text.py
python -m pytest
```

Read `lessons/68_save_vocabulary.md` through
`lessons/72_text_inference_cli.md`. The tiny included corpus makes
this an **educational example**, not a high-quality pretrained text
generator. Unseen prompt characters are rejected rather than
silently remapped. The combined checkpoint is for inference;
optimizer and RNG resume state are covered in earlier lessons.

## Reproducible training and validation (commits 076–080)

We now estimate training/validation loss on **repeatably sampled**
held-out batches, monitor loss at regular steps, retain the lowest
validation-loss model, and save it alongside the tokenizer.

Run a training experiment from the repository root:

```bash
python -m rebuild_gpt.train_cli \
  --corpus data/tiny_story.txt --steps 60 \
  --eval-every 10 --eval-batches 4 --seed 7 \
  --output out/char_gpt.pt \
  --best-output out/best_char_gpt.pt \
  --history-file out/training_history.json

python -m rebuild_gpt.text_cli \
  --checkpoint out/best_char_gpt.pt --prompt "a" \
  --new-tokens 80 --temperature 0.8 --top-k 5 --seed 13
```

The JSON log records config, seed and per-step training/validation
loss estimates. `--restore-best` puts the best observed weights in
`--output` as well as `--best-output`. Both checkpoint files contain
the model's exact character vocabulary for inference.

Follow `lessons/73_repeatable_evaluation.md` through
`lessons/77_reproducible_training_cli.md`, or run:

```bash
python examples/70_evaluate_splits.py
python examples/71_training_history.py
python examples/72_best_model_snapshot.py
python examples/73_restore_best_text_gpt.py
python examples/74_monitored_training.py
python -m pytest
```

**Limitation:** Reused seeded mini-batches reduce comparison noise,
but they are not a complete validation-set evaluation. The included
tiny story is educational; better held-out loss does not establish
useful or coherent language generation.
