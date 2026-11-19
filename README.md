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

## Training stability and full-split evaluation (commits 081–085)

These lessons add optional **global gradient clipping**, nanoGPT-style
**linear warmup and cosine learning-rate decay**, and an alternative
validation method that scores **every next-token transition once** in
consecutive nonoverlapping chunks. Sampled evaluation remains the
default, and existing training scripts remain backward compatible.

```bash
python -m rebuild_gpt.train_cli \
  --corpus data/tiny_story.txt --steps 60 \
  --eval-every 10 --validation-mode full \
  --grad-clip 1.0 \
  --warmup-steps 5 --lr-decay-steps 50 --min-lr 0.002 \
  --output out/char_gpt.pt \
  --best-output out/best_char_gpt.pt \
  --history-file out/training_history.json \
  --restore-best
```

Run `examples/75_gradient_clipping.py` through
`examples/79_full_training_controls.py`, read
`lessons/78_gradient_clipping.md` through
`lessons/82_full_training_controls.md`, and run `python -m pytest`.

**Caution:** Exhaustive scoring here means exhaustive next-token
targets under a **chunked context reset policy**. It does not mean a
token-by-token sliding-window score or a representative benchmark
on a large language corpus. Better validation estimates do not
guarantee coherent generated text on the tiny included story.

## Gradient accumulation and nanoGPT-style AdamW (commits 086–090)

You can now accumulate **equally sized** microbatch gradients over
multiple forward/backward passes, then clip and update **once**.
Grouped AdamW uses nanoGPT's parameter-rank rule: matrices (2D+)
receive configured weight decay, while biases and normalization
vectors do not. Tied token/output embeddings are included once.

Both behaviors are opt-in so old lessons keep their original results.

```bash
python -m rebuild_gpt.train_cli \
  --corpus data/tiny_story.txt --steps 30 \
  --gradient-accumulation-steps 3 \
  --grouped-adamw --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.95 \
  --grad-clip 1.0 \
  --warmup-steps 3 --lr-decay-steps 25 --min-lr 0.002 \
  --eval-every 10 \
  --output out/char_gpt.pt \
  --best-output out/best_char_gpt.pt \
  --history-file out/training_history.json
```

`--steps` counts **optimizer updates**, not microbatches.
The effective token positions per update are `batch_size *
block_size * gradient_accumulation_steps` (sampled with replacement).
The run metadata records optimizer group settings and accumulation.

Run `examples/80_gradient_accumulation.py` through
`examples/84_optimizer_training_cli.py`, study
`lessons/83_gradient_accumulation.md` through
`lessons/87_optimizer_training_cli.md`, and check `python -m pytest`.

**Not yet the full nanoGPT training script:** This remains a
single-process educational CPU workflow without mixed precision,
distributed data parallelism, or complete optimizer checkpoint
resume support for grouped optimizer configurations.

## Resumable CPU text training (commits 091–095)

The character-GPT trainer now distinguishes inference checkpoints from
**training-state** checkpoints, including optimizer moments, current model
weights, batch/CPU RNG states, validation progress, and a corpus fingerprint.

```bash
# Save state after two optimizer updates
python -m rebuild_gpt.train_cli --steps 2 --eval-every 2 \
  --training-state out/train_state.pt

# Resume to step four using identical hyperparameters
python -m rebuild_gpt.train_cli --steps 4 --eval-every 2 \
  --resume-from out/train_state.pt --training-state out/train_state.pt
```

`--steps` is always the **total** optimizer-update target. Resuming
with different data, tokenizer, model configuration or training controls
fails rather than silently starting a different experiment. Keep
training-state files trusted: they are loaded via
`torch.load(..., weights_only=True)`.

The resumable training state is now written via a same-directory temporary
file and `os.replace`, protecting the previous checkpoint from incomplete
writes. To prove interrupted and uninterrupted training agree, run:

```bash
python examples/85_resume_text_gpt.py
python -m pytest tests/test_text_resume.py tests/test_best_resume.py \
  tests/test_text_training_resume.py tests/test_cli_resume.py -q
```

These lessons currently target **CPU** training; full CUDA RNG and
mixed-precision scaler persistence remain future work.

## CPU/CUDA training (commits 096–100)

The character-level GPT trainer supports `--device cpu|cuda|auto`.
The default is `cpu`. `cuda` requires PyTorch CUDA support;
`auto` selects CUDA if available and otherwise uses CPU.

```bash
python examples/86_device_training.py --device cpu
python -m rebuild_gpt.train_cli --device auto --steps 2
python -m pytest tests/test_devices.py tests/test_device_cli.py \
  tests/test_text_training_devices.py tests/test_evaluation.py \
  tests/test_text_resume.py -q
```

The original CPU data stream stays on CPU. Sampled training and
evaluation windows transfer to the model device. CUDA training
checkpoints retain CUDA RNG state for dropout, alongside the CPU RNG
and batch-sampling generator. Resume requires compatible training
controls and the same selected backend. Mixed precision and
distributed training are not implemented yet.

## Mixed precision training (commits 101–105)

These lessons add explicit FP32 (default), BF16 and CUDA FP16 computation.
Model parameters and AdamW state remain FP32; autocast wraps forward
passes and validation. CUDA FP16 uses dynamic loss scaling, unscales
before clipping, and stores its scaler state for resumable training.

```bash
python -m rebuild_gpt.train_cli --device cpu --precision bf16 --steps 2
python examples/87_mixed_precision.py --device cpu --precision bf16
# Requires a supported CUDA GPU:
python -m rebuild_gpt.train_cli --device cuda --precision fp16 --steps 2
python -m pytest tests/test_precision.py tests/test_amp_steps.py \
  tests/test_mixed_precision_training.py tests/test_precision_cli.py -q
```

Read `lessons/98_precision_policy.md` through
`lessons/102_mixed_precision_cli.md`. CUDA FP16 behavior and performance
still require tests on GPU hardware. Distributed training is a later goal.

## Optimized scaled-dot-product attention (commits 106–110)

The original fused-manual QKV path is preserved for education and
checkpoint parity, with a new SDPA computation option:

```bash
python -m rebuild_gpt.train_cli --attention-impl sdpa --steps 2
python examples/88_attention_benchmark.py --iterations 10 --warmup 3
python -m pytest tests/test_sdpa_attention.py tests/test_sdpa_gpt.py \
  tests/test_sdpa_training.py tests/test_attention_benchmark.py -q
```

The manual and SDPA modules use **the same parameter names and shapes**.
The attention calculation may select different kernels and differ
slightly numerically. Benchmark results depend on CPU/GPU, precision,
sequence length and warmup; speedups are not guaranteed.

Lessons: `103_sdpa_attention.md` through `107_sdpa_benchmark.md`.
GitHub Actions now runs CPU tests; GPU coverage still needs a runner
with CUDA hardware.

## Benchmarking and CI checks (commits 111–115)

GitHub Actions verifies Python syntax and runs the full CPU pytest suite.
When tests fail, a JUnit XML report is attached to the workflow run.

```bash
python -m compileall -q src tests examples
python -m pytest -q
python examples/88_attention_benchmark.py --iterations 5
python examples/89_training_benchmark.py --iterations 5
```

The training benchmark includes forward, backward and optimizer steps;
the inference benchmark measures only forward-pass latency. Both use
warmup and, on CUDA, synchronization. Neither benchmark assumes that
SDPA is always faster. See `lessons/108_ci_regression.md` through
`lessons/112_ci_quality_gates.md` for the corresponding lessons.

## Memory and profiling (commits 116–120)

These lessons distinguish **estimated model storage**, **measured CUDA
allocator peaks**, and **latency or operator-level profiles**. A CPU-only
machine can run all educational examples except measured CUDA memory.

```bash
python -m pytest tests/test_memory.py tests/test_cuda_memory.py \
  tests/test_memory_comparison.py tests/test_context_sweep.py \
  tests/test_torch_profiler.py -q

# Model/AdamW persistent tensor estimate, optionally measure CUDA peak
python examples/90_memory_comparison.py --device cpu

# Time multiple context lengths and save CSV
python examples/91_context_sweep.py --lengths 4,8,16 --csv out/context_sweep.csv

# Export an operator-level timeline for Chrome trace viewers
python examples/92_torch_profiler.py --device cpu \
  --output out/profiles/tiny_gpt.json
```

For a CUDA machine, use `--device cuda` on supported examples.
Profiler instrumentation adds overhead; do not use profiled timings as
unbiased throughput benchmarks. Traces may contain operation shapes,
names and workload metadata, so review them before sharing.

Read `lessons/113_memory_budget.md` through
`lessons/117_operator_profiler.md` in order.

## Activation memory and checkpointing (commits 121–125)

Learn why explicit attention scores scale as `[B,H,T,T]` and how
non-reentrant activation checkpointing can trade extra backward
compute for fewer saved intermediate tensors.

```bash
python -m pytest tests/test_activation_memory.py \
  tests/test_activation_checkpointing.py tests/test_checkpointed_gpt.py \
  tests/test_checkpointed_training.py tests/test_checkpoint_benchmark.py -q

# Train a tiny GPT with block checkpointing enabled
python -m rebuild_gpt.train_cli --steps 2 \
  --attention-impl sdpa --activation-checkpointing

# Compare full training-step latency, optionally CUDA allocator deltas
python examples/93_activation_checkpoint_benchmark.py --device cpu
```

Activation checkpointing changes **training computation**, not model
parameter names or inference checkpoint format. The flag is recorded
in resumable training controls when active; mismatched resumes are
rejected. CPU tests validate numerical gradients, dropout replay and
CLI behavior; real CUDA memory savings require supported GPU hardware.

Read `lessons/118_attention_matrix_scaling.md` through
`lessons/122_checkpoint_benchmark.md` for the five exercises.

## Selective activation checkpointing (commits 126–130)

Checkpoint only every Nth transformer block, or preserve the previous
all-or-none setting. The new policy does not change GPT weights,
inference checkpoint names or model architecture.

```bash
# Every second block of a four-block model:
python -m rebuild_gpt.train_cli --steps 2 --n-layer 4 \
  --checkpoint-every-n 2 --attention-impl sdpa

# Compare off, selective, and all (CPU latency; CUDA VRAM if available):
python examples/94_selective_checkpoint_benchmark.py \
  --device cpu --n-layer 4 --checkpoint-every-n 2

python -m pytest tests/test_checkpoint_policy.py \
  tests/test_selective_stack.py tests/test_selective_checkpoint_parity.py \
  tests/test_selective_training_cli.py \
  tests/test_selective_checkpoint_benchmark.py -q
```

The selected interval is validated before training, reported by the
CLI, and recorded in resumable training controls when active.
Previous default-off training checkpoints remain compatible.
See lessons `123_select_checkpoint_policy.md` through
`127_selective_benchmark.md` for objectives and experiments.

## Batch, depth and context scaling (commits 131–135)

These lessons separate **analytical score-matrix storage** from
**measured training-step latency**. The workload estimator tracks
effective tokens per optimizer update, and the grid generator produces
bounded experiment configurations with CSV output.

```bash
# Predict attention score sizes; no model/GPU needed:
python examples/95_scaling_grid.py --batch-sizes 1,2 \
  --contexts 4,8,16 --depths 1,2,4 --csv out/workload_grid.csv

# Measure real training steps and record reproducibility metadata:
python examples/96_depth_batch_benchmark.py \
  --batch-sizes 1,2 --depths 1,2 --context 8 \
  --seed 77 --csv out/training_times.csv \
  --json out/training_report.json

python -m pytest tests/test_workload_scaling.py \
  tests/test_scaling_grid.py tests/test_scaling_grid_cli.py \
  tests/test_depth_batch_benchmark.py tests/test_benchmark_report.py -q
```

Both CPU and CUDA training benchmarks reuse synthetic microbatches;
throughput excludes data loading. CSV timings and JSON metadata are
measurement artifacts, not claims of hardware-independent speedups.
See lessons 128–132 in `lessons/`.

## Logical-batch microbatching (commits 136–140)

The project now distinguishes the **logical batch** (all examples
contributing to one update) from the **microbatch** (examples processed
together in one forward/backward pass). Uneven last chunks are
weighted by target count, so every next-token target contributes
equally.

```bash
# Five sequences per optimizer update, split (2,2,1):
python -m rebuild_gpt.train_cli --steps 2 \
  --batch-size 5 --microbatch-size 2

# Compare the same logical batch across chunk sizes:
python examples/97_microbatch_benchmark.py \
  --logical-batch-size 6 --microbatch-sizes 1,2,3,6 \
  --csv out/microbatch.csv --json out/microbatch.json

python -m pytest tests/test_microbatches.py \
  tests/test_uneven_microbatch_gradients.py \
  tests/test_logical_batch_step.py \
  tests/test_microbatch_benchmark.py \
  tests/test_text_microbatch_cli.py -q
```

Older `--gradient-accumulation-steps` behavior is unchanged. It
samples that many *independent* batches of size `--batch-size`.
The new `--microbatch-size` divides one batch and cannot be used
with accumulation steps >1. The new mode's setting is saved in
training-state controls; default-mode checkpoints remain compatible.
See lessons 133–137 in `lessons/`.

## Streamed microbatches and learning outcomes (commits 141–145)

The text training path now keeps the *logical* training batch on CPU
and transfers only the current microbatch to the compute device before
its forward/backward pass. After processing all weighted chunks, it
performs one optimizer update.

```bash
# Existing CLI flag now uses streaming internally:
python -m rebuild_gpt.train_cli --steps 2 \
  --batch-size 5 --microbatch-size 2

# Compare eager full-batch transfers vs streamed transfers:
python examples/97_microbatch_benchmark.py \
  --logical-batch-size 6 --microbatch-sizes 1,2,6 \
  --transfer-mode streamed --csv out/streamed.csv

# Compare what the model learns at fixed logical batch/seed:
python examples/98_compare_microbatch_learning.py \
  --steps 3 --batch-size 5 --microbatch-sizes 1,2,5 \
  --json out/microbatch_learning.json

python -m pytest tests/test_weighted_microbatch_iterator.py \
  tests/test_streamed_microbatch_step.py \
  tests/test_streamed_text_training.py \
  tests/test_streamed_microbatch_benchmark.py \
  tests/test_microbatch_learning_comparison.py -q
```

Loss and gradient equivalence are verified within floating-point
tolerances; no particular speedup or VRAM reduction is assumed.
On CPU, tests exercise correctness and report GPU memory as unavailable.
Actual CUDA peaks must be measured on supported hardware.
See lessons 138–142 in `lessons/`.

## CPU-to-CUDA transfer pipeline (commits 146–150)

The streamed logical-batch trainer can optionally stage one whole CPU
batch in pinned host memory, then copy individual microbatches to CUDA.
Pinned and nonblocking copies are **opt-in**: CPU training continues to
use the original blocking path, with no CUDA initialization required.

```bash
# Original behavior:
python -m rebuild_gpt.train_cli --steps 2 \
  --batch-size 5 --microbatch-size 2

# Requires a CUDA-enabled PyTorch installation and compatible GPU:
python -m rebuild_gpt.train_cli --device cuda \
  --steps 2 --batch-size 5 --microbatch-size 2 \
  --pin-memory --non-blocking-transfer

# CPU reports sampling only; CUDA compares actual staging+copy variants:
python examples/99_host_transfer_benchmark.py --device cpu
python examples/99_host_transfer_benchmark.py --device cuda \
  --iterations 20 --warmup 5 --csv out/transfers.csv

python -m pytest tests/test_pinned_batch.py \
  tests/test_microbatch_transfer.py tests/test_streamed_transfer_controls.py \
  tests/test_host_transfer_benchmark.py \
  tests/test_transfer_training_cli.py -q
```

Nonblocking copies use the **current CUDA stream**, without transfer/
compute overlap on a separate stream. Pinning incurs CPU work; neither
performance nor memory savings are guaranteed. The CLI records the
transfer flags and checks them when resuming training-state checkpoints.
See `lessons/143_pinned_host_memory.md` through
`lessons/147_transfer_training_controls.md`.

## Pipeline bottlenecks and scaling (commits 151–155)

Separate the measured cost of sampling, CPU-to-device staging and
model forward/backward/AdamW. The timed pipeline uses deliberate CUDA
synchronization between stages so asynchronous launches are attributed
to the correct work; it does **not** measure transfer/compute overlap.

```bash
# Compare identical token-window starts with two sampling methods:
python examples/100_sampling_benchmark.py --batch-size 64 --block-size 32

# Profile complete training pipeline by stage:
python examples/101_training_pipeline_breakdown.py --device cpu \
  --csv out/pipeline_one.csv --json out/pipeline_one.json

# Sweep batch size/depth and explain the largest observed stages:
python examples/102_pipeline_scaling_sweep.py --device cpu \
  --batch-sizes 1,2,4 --depths 1,2 \
  --csv out/pipeline_sweep.csv --json out/pipeline_sweep.json
python examples/103_analyze_pipeline.py \
  --input out/pipeline_sweep.csv --output out/pipeline_analysis.json

python -m pytest tests/test_pipeline_timing.py \
  tests/test_fixed_window_gather.py tests/test_sampling_benchmark.py \
  tests/test_training_pipeline_breakdown.py \
  tests/test_pipeline_scaling_sweep.py tests/test_pipeline_analysis.py -q
```

The analyzer's stage fractions are calculated using the **sum of
stage medians**, not the median end-to-end time. Bottleneck labels
identify investigation candidates only, and neither CUDA speedups
nor CPU timing improvements are guaranteed. See lessons 148–152.

## Cached token-window sampling (commits 156–160)

An opt-in sampler builds a zero-copy `unfold` view of the training
token stream once, then gathers batches from randomly selected rows.
Repeated draws preserve the same token IDs and RNG progression as the
original indexed sampler. **The default trainer is unchanged.**

```bash
# Run actual character-GPT training with the cached sampler:
python -m rebuild_gpt.train_cli --steps 3 --sampling-backend cached

# Compare sampling-only CPU latency over several contexts:
python examples/104_cached_sampler_benchmark.py \
  --batch-sizes 4,16 --contexts 4,16 \
  --csv out/cached_sampler.csv --json out/cached_sampler.json

# Then compare sampling's contribution to a complete training step:
python examples/101_training_pipeline_breakdown.py \
  --device cpu --sampling-backend indexed --csv out/indexed_pipeline.csv
python examples/101_training_pipeline_breakdown.py \
  --device cpu --sampling-backend cached --csv out/cached_pipeline.csv

python -m pytest tests/test_unfold_gather.py \
  tests/test_window_sampler.py tests/test_cached_text_training.py \
  tests/test_cached_sampling_benchmark.py \
  tests/test_cached_pipeline_profiles.py -q
```

The sampler choice is written to the CLI history and to resumable
training controls when nondefault; old indexed-mode checkpoints
remain compatible. No speedup is assumed. CPU and GPU execution can
favor different strategies. See lessons 153–157 in `lessons/`.

## Reusable sampled evaluation batches (commits 161–165)

The trainer can now prepare **fixed CPU sampled validation batches
once** and reuse them at each evaluation step. Since the previous
regenerated evaluator already used the same seed each time, this
optional optimization preserves evaluation examples and model updates.

```bash
# Train with an opt-in prepared evaluation cache:
python -m rebuild_gpt.train_cli --steps 5 --eval-every 1 \
  --validation-mode sampled --eval-batch-backend cached

# Benchmark one-time cache setup vs repeated evaluation:
python examples/105_cached_eval_benchmark.py \
  --eval-batches 1,2,4 --csv out/eval_cache.csv \
  --json out/eval_cache.json

python -m pytest tests/test_eval_cache.py \
  tests/test_prepared_evaluation.py tests/test_cached_eval_training.py \
  tests/test_cached_eval_benchmark.py tests/test_cached_eval_cli.py -q
```

The cache retains input **and** target CPU tensors and can increase
host RAM usage. It still transfers each evaluation batch to the
model's compute device; a performance improvement is not guaranteed.
The full-corpus validation mode doesn't use this cache.
Nondefault evaluation policy is recorded in resumable training
controls; existing default-mode checkpoints remain compatible.
See lessons 158–162 in `lessons/`.

## Budgeted evaluation caching and break-even (commits 167–171)

The opt-in sampled evaluation cache can now be bounded by a
conservative CPU memory budget. The budget covers the retained
logical input/target bytes across both evaluation splits; it does
**not** measure total process RSS or GPU VRAM.

```bash
# Bound the cache and record its estimated size in the training report:
python -m rebuild_gpt.train_cli --steps 5 --eval-every 1 \
  --eval-batch-backend cached --eval-cache-max-mib 16

# Measure setup overhead, repeated evaluation latency and break-even:
python examples/105_cached_eval_benchmark.py \
  --eval-batches 1,2,4 --eval-cache-max-mib 16 \
  --csv out/eval_budget.csv --json out/eval_budget.json

python -m pytest tests/test_eval_cache_budget.py \
  tests/test_eval_cache_limits.py tests/test_eval_cache_training_budget.py \
  tests/test_eval_cache_budget_cli.py tests/test_cache_economics.py \
  tests/test_cached_eval_benchmark.py -q
```

The estimate counts both logical input and target elements. In our
sampler they can share a base allocation, making this estimate
conservative. The cap is optional and recorded in resumable
training-state controls when enabled; a mismatch stops resumption.
An oversized plan fails *before* sampling either split.

Break-even calculations from benchmark medians are illustrative,
not guaranteed speedups. See lessons 163–167 in `lessons/`.

## Compact index-only validation plans (commits 173–177)

Sampled validation can now cache only the **random start positions**
instead of holding complete input and target tensors on CPU. This
reduces retained plan data for longer contexts while keeping the
original corpus alive and regathering selected windows on demand.

```bash
# Index-only validation; budget limits retained index storage:
python -m rebuild_gpt.train_cli --steps 5 --eval-every 1 \
  --eval-batch-backend indices --eval-cache-max-mib 1

# Compare regeneration, full token caching, and index-only caching:
python examples/106_index_eval_benchmark.py \
  --batch-size 8 --context 32 --eval-batches 4 \
  --csv out/index_eval.csv --json out/index_eval.json

python -m pytest tests/test_eval_index_plan.py \
  tests/test_eval_index_budget.py tests/test_index_plan_evaluation.py \
  tests/test_index_eval_training.py tests/test_index_eval_cli.py \
  tests/test_index_eval_benchmark.py -q
```

The three backends use identical seeded validation examples and
preserve model weights and checkpoint controls. The retained-index
budget excludes corpus storage and temporary gathered windows.
Benchmark medians are evaluation-only, not complete training
latency; no performance speedup is promised. See lessons 168–172.

## Sampled validation dispersion and paired comparisons (commits 178–182)

This milestone adds descriptive statistics for individual sampled
evaluation losses, including sample standard deviation and standard
error. It also makes the per-batch losses observable for all three
existing validation backends without changing any training behavior.

```bash
# Sample the same held-out text with several validation seeds:
python examples/107_validation_seed_sweep.py \
  --corpus data/tiny_story.txt --seed-count 4 \
  --csv out/seed_sweep.csv --json out/seed_sweep.json

# Compare an untrained GPT with its weights after one AdamW update,
# using identical validation windows for both model versions:
python examples/108_paired_validation.py \
  --corpus data/tiny_story.txt --seed-count 3 \
  --csv out/paired.csv --json out/paired.json

python -m pytest tests/test_evaluation_stats.py \
  tests/test_evaluation_samples.py tests/test_seed_sensitivity.py \
  tests/test_validation_seed_sweep.py tests/test_paired_evaluation.py \
  tests/test_paired_validation_cli.py -q
```

These comparisons describe **sampled held-out loss**, not a calibrated
confidence interval or statistical significance. Repeated RNG seeds
can still select overlapping windows from one corpus. The standard
training CLI and checkpoint selection policy remain unchanged.
See lessons 173–177 in `lessons/`.

## Paired validation diagnostics and saved checkpoint comparisons (commits 183–187)

We can now inspect *per-batch* candidate-minus-baseline losses and
count how often sampled validation batches or seed means improve,
worsen or tie. These are **descriptive** measures: token windows
overlap, so sign counts cannot be interpreted as independent
Bernoulli trials or a statistical significance test.

```bash
# Compare existing self-contained GPT inference checkpoints on the
# same held-out text and validation seeds:
python examples/109_compare_saved_checkpoints.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt --context 8 \
  --csv out/paired_checkpoints.csv --json out/paired_checkpoints.json

# Sweep the number of sampled validation batches:
python examples/110_paired_budget_sweep.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt \
  --eval-batch-counts 1,2,4,8 --seed-count 3 \
  --csv out/paired_sweep.csv --json out/paired_sweep.json

python -m pytest tests/test_paired_batch_evaluation.py \
  tests/test_paired_diagnostics.py tests/test_checkpoint_comparison.py \
  tests/test_saved_checkpoint_comparison_cli.py \
  tests/test_paired_budget_sweep.py -q
```

Both checkpoints must store identical character-to-ID vocabulary
order. The comparison uses the same contiguous 20% held-out split
and identical seeded token windows. The sweep increases sample count
without retraining or claiming any generalization confidence bound.
See lessons 178–182 in `lessons/`.

## Overlap-aware paired validation (commits 188–192)

Sampled held-out windows can repeat starts or overlap in token
positions. The new diagnostics count both, while an opt-in
`disjoint` policy samples nonoverlapping fixed T+1-token tiles
without replacement **within each seed's plan**.

```bash
# Original policy stays the default; disjoint is explicit:
python examples/109_compare_saved_checkpoints.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt \
  --sampling-policy disjoint --context 8 --eval-batches 3 \
  --csv out/disjoint_checkpoints.csv --json out/disjoint_checkpoints.json

# Compare validation sample counts under both policies:
python examples/110_paired_budget_sweep.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt --sampling-policy both \
  --eval-batch-counts 1,2,4 \
  --csv out/policy_sweep.csv --json out/policy_sweep.json

python -m pytest tests/test_window_overlap.py \
  tests/test_disjoint_evaluation.py tests/test_disjoint_paired_validation.py \
  tests/test_disjoint_checkpoint_cli.py tests/test_policy_budget_sweep.py -q
```

Disjoint tiling changes the distribution of possible starts; it
does **not** make one corpus independent observations. Different
seeds may choose the same tile, and a disjoint plan requires enough
tiles for all requested windows. No statistical significance or
generalization guarantee is implied. See lessons 183–187.

## Phase-shifted disjoint validation (commits 193–197)

The original sampled validation draws starts with replacement.
Fixed-origin disjoint sampling uses nonoverlapping T+1-token tiles,
but only from starts aligned to token zero. The new opt-in
`shifted` policy draws a **seeded feasible lattice phase** before
selecting distinct nonoverlapping tiles.

```bash
# Compare saved models using seed-shifted disjoint windows:
python examples/109_compare_saved_checkpoints.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt --sampling-policy shifted \
  --csv out/shifted_comparison.csv --json out/shifted_comparison.json

# Compare all three sampling distributions at multiple budgets:
python examples/110_paired_budget_sweep.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt --sampling-policy all \
  --eval-batch-counts 1,2,4 \
  --csv out/phase_sweep.csv --json out/phase_sweep.json

python -m pytest tests/test_shifted_evaluation.py \
  tests/test_lattice_diagnostics.py tests/test_shifted_paired_validation.py \
  tests/test_shifted_checkpoint_cli.py tests/test_shifted_budget_sweep.py -q
```

Results record exact phases, sample budgets and overlap counts.
Phase feasibility can change when the requested window count
changes, so shifted budget sweeps **need not preserve sampled
prefixes**. The shifted policy changes the sampling distribution
and cannot by itself justify independence or unbiasedness. See
lessons 188–192 in `lessons/`.

## Global validation coverage across seeds (commits 199–203)

A nonoverlapping validation plan may have zero repeated token
positions *within one seed* while sharing the same windows with
other seeds. We now separate within-seed and between-seed
repetitions, and report the true global union of touched tokens.

```bash
# Show global coverage when comparing saved GPT checkpoints:
python examples/109_compare_saved_checkpoints.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt --sampling-policy shifted \
  --json out/paired_global_coverage.json

# Compare global token coverage over a larger number of seeds:
python examples/111_seed_coverage_sweep.py \
  --corpus-tokens 2048 --batch-size 4 --context 16 \
  --seed-counts 1,2,4,8 --sampling-policy all \
  --csv out/seed_coverage.csv --json out/seed_coverage.json

python -m pytest tests/test_cross_seed_overlap.py \
  tests/test_paired_cross_seed_overlap.py \
  tests/test_cross_seed_checkpoint_cli.py \
  tests/test_cross_seed_budget_sweep.py \
  tests/test_seed_coverage_sweep.py -q
```

The previous per-seed overlap fields remain available. In the
policy sweep, the legacy summed unique-coverage field is **not**
a globally deduplicated count; use the new
`global_unique_corpus_token_positions` field instead.

These are reproducible coverage diagnostics, not statistical
independence tests or generalization guarantees. See lessons
193–197 in `lessons/`.

## Marginal validation coverage and retrospective goals (commits 204–208)

The validation seed experiments now report **new token positions
added by each seed**, positions already touched by earlier seeds,
and overlap within each seed. The per-seed marginal gains sum to
the globally distinct corpus positions touched by the full plan.

An optional `--coverage-target` reports the first *preselected*
seed prefix reaching a requested fraction of corpus positions,
or the remaining shortfall if the target is not reached. It never
adapts the selected seeds, alters training, or claims statistical
confidence.

```bash
# Synthetic token-position coverage and target crossing:
python examples/111_seed_coverage_sweep.py \
  --corpus-tokens 1024 --seed-counts 1,2,4,8 \
  --coverage-target 0.5 --sampling-policy all \
  --json out/coverage_targets.json

# Equivalent model-free coverage analysis on held-out UTF-8 text:
python examples/112_heldout_seed_coverage.py \
  --corpus data/tiny_story.txt --batch-size 4 --context 8 \
  --seed-counts 1,2,4 --sampling-policy all \
  --csv out/heldout_coverage.csv --json out/heldout_coverage.json

python -m pytest tests/test_coverage_growth.py \
  tests/test_marginal_coverage_report.py tests/test_coverage_targets.py \
  tests/test_seed_coverage_targets_cli.py \
  tests/test_heldout_seed_coverage.py -q
```

These statistics describe **coverage of one underlying corpus**.
They are not effective sample sizes, model performance metrics,
calibrated confidence bounds, or statistical stopping rules.
See lessons 198–202 in `lessons/`.

## Regional validation coverage and blind-spot diagnostics (commits 209–213)

An evaluation may touch many held-out token positions while
concentrating almost all its windows in a few corpus regions.
The new positional profiler divides validation text into
equally spaced regions and reports unique coverage, repeated
visits, and coverage fraction for each.

```bash
python examples/112_heldout_seed_coverage.py \
  --corpus data/tiny_story.txt --seed-counts 1,2,4 \
  --sampling-policy all --position-bins 8 \
  --minimum-bin-coverage 0.5 \
  --csv out/coverage_summary.csv \
  --position-bin-csv out/coverage_regions.csv \
  --json out/coverage_details.json

python -m pytest tests/test_position_coverage.py \
  tests/test_position_growth.py tests/test_position_targets.py \
  tests/test_heldout_position_cli.py \
  tests/test_heldout_position_csv.py -q
```

JSON preserves per-region marginal coverage gains and reports
whether every region met the optional threshold. The new
long-format positional CSV can be plotted by bin index.
All samples are still drawn from a **single** held-out text:
coverage does not establish independence, unbiasedness or
model performance. See lessons 203–207 in `lessons/`.

## Region-balanced validation sampling (commits 214–218)

A new opt-in `balanced` policy samples a nearly equal number of
window **starts** in every eligible-start region of held-out text.
It reduces the chance of unsampled start regions, but it **changes
the sampling distribution**, permits overlapping windows, and
does not guarantee balanced token coverage or unbiased loss.

```bash
# Compare two saved GPTs on identical balanced start plans:
python examples/109_compare_saved_checkpoints.py \
  --baseline out/model_before.pt --candidate out/model_after.pt \
  --corpus data/tiny_story.txt \
  --sampling-policy balanced --sampling-regions 4 \
  --csv out/balanced_checkpoints.csv --json out/balanced_checkpoints.json

# Compare three existing samplers plus balanced on held-out text:
python examples/112_heldout_seed_coverage.py \
  --corpus data/tiny_story.txt --sampling-policy extended \
  --sampling-regions 4 --seed-counts 1,2,4 \
  --position-bins 8 --csv out/four_policies.csv \
  --json out/four_policies.json

python -m pytest tests/test_balanced_evaluation.py \
  tests/test_start_region_diagnostics.py \
  tests/test_balanced_paired_evaluation.py \
  tests/test_balanced_checkpoint_cli.py \
  tests/test_balanced_heldout_coverage.py -q
```

Paired evaluation and checkpoint comparison still default to
`replacement`. The held-out coverage CLI's legacy `all` still
means three policies; `extended` explicitly selects four.
New reports distinguish start counts from actual token coverage.
See lessons 208–212 in `lessons/`.
