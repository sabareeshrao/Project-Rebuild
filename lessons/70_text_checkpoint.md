# Lesson 70: an inference checkpoint needs both weights and vocabulary

Our earlier model checkpoint stored the GPT configuration and weights,
but not the character vocabulary. To generate text in another process,
we also need the **exact ID order** used during training.

The new self-contained checkpoint stores plain values:

```text
format_version | GPTConfig fields | ordered character vocab | state_dict
```

On loading, it verifies the checkpoint version, reconstructs the
tokenizer, checks the model-vocabulary size agreement, strictly loads
all parameter keys/shapes, and selects evaluation mode.

The loader uses CPU mapping and `weights_only=True` when
deserializing PyTorch tensors. Only open checkpoints from sources
you trust.

Run: `python examples/67_text_checkpoint.py`.

**Experiment:** Modify the saved vocabulary or remove one weight.
What validation failure do you see?

**Checkpoint:** Why is a checkpoint with both tensors and token ID
mapping safer than attempting to rebuild the vocabulary from an
unrelated input prompt?
