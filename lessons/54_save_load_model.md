# Lesson 54: save and reload the same model

A useful checkpoint stores **architecture configuration** and the
model's **named tensors**. We deliberately save plain dictionaries
rather than pickling entire Python model objects.

```text
GPTConfig + state_dict -> torch.save -> checkpoint.pt
checkpoint.pt -> torch.load(weights_only=True) -> GPTConfig
                                                 -> new GPTLanguageModel
                                                 -> strict load_state_dict
```

Our loader uses `map_location="cpu"`, `weights_only=True`, and
`strict=True`. It rejects unknown checkpoint format versions and
missing or mismatched parameter keys. It sets evaluation mode before
returning the restored model.

Run: `python examples/54_save_load_model.py`.

**Experiment:** Edit the saved checkpoint to remove one tensor name.
Why does strict loading fail?

**Checkpoint:** Why do we reconstruct the architecture before
loading weights, and why is `state_dict` more portable than saving
a whole Python model object?

**Security:** Only load checkpoint files from trusted sources.
`weights_only=True` narrows deserialization risk, but does not
make arbitrary untrusted files harmless.

This checkpoint format is ours, not the original nanoGPT's training
checkpoint format. Training-optimizer state comes next.
