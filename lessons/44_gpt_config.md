# Lesson 44: model hyperparameters should agree

A configuration stores the architecture in one place. Important
choices are:

- `vocab_size`: how many output token IDs are possible.
- `block_size`: maximum positions in an input context window.
- `n_embd`: feature width of each token vector.
- `n_head`: how many independent attention heads.
- `n_layer`: how many transformer blocks are stacked.
- `dropout`: regularization rate.

Each head uses width `n_embd / n_head`, so **n_embd must be divisible
by n_head** in this implementation. A frozen dataclass prevents
accidental edits after constructing a model, and explicit validation
produces useful errors before expensive training begins.

Run: `python examples/44_gpt_config.py`.

**Experiment:** Set `n_embd=10` and `n_head=3`. Explain the error.

**Checkpoint:** Which hyperparameter controls model *depth*, and which
controls the number of features per token?
