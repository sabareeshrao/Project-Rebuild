# Lesson 82: run an improved and reproducible GPT training experiment

The training CLI now exposes our three new concepts together:

- `--grad-clip 1.0` caps the global norm of gradients.
- `--warmup-steps 3 --lr-decay-steps 25 --min-lr 0.002`
  enables nanoGPT-style warmup followed by cosine LR decay.
- `--validation-mode full` evaluates **every next-token transition**
  on train and validation splits using nonoverlapping chunks.

The other CLI settings still select the corpus, model dimensions,
evaluation interval, seed, checkpoint paths, and JSON history output.
`--validation-mode sampled` remains the default for speed.

**Important qualification:** Full mode is exhaustive over targets,
but not an all-prefix sliding-window perplexity benchmark. Context is
reset at `block_size` boundaries, and the last chunk may be shorter.
Unlike sampling a few random batches, it doesn't omit or duplicate
targets within each split.

Run: `python examples/79_full_training_controls.py`.

**Experiment:** Train with identical seeds using `sampled` and
`full` validation. Do the optimizer updates stay identical when
checkpoint selection does not restore an earlier snapshot? Do
the selected best steps always match?

**Checkpoint:** Why could checkpoint selection change with a
different validation policy, even when every training update is
identical?

**Next:** More realistic data preprocessing, gradient accumulation,
precision choices, and AdamW parameter-group configuration.
