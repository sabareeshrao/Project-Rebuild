# Lesson 288 — Inspect the complete train-to-sample artifact set

Our capstone has three output artifacts:

- `model.pt`: model weights, configuration and tokenizer.
- `sample.txt`: generated text from a fixed sampling seed.
- `run.json`: training settings, metrics, sample and SHA-256 digests.

Before loading any checkpoint, compare the saved file bytes with the
manifest. This catches missing or altered model and corpus files and
a changed generated sample. It does **not** execute PyTorch.

```text
capstone/run.json -> validate report and digests
  |-> corpus text  -> SHA-256 match
  |-> model.pt     -> SHA-256 match
  +-> sample.txt   -> exact text match
```

Run `python examples/120_capstone_cli.py --steps 4` first, then
`python examples/181_capstone_artifact_integrity.py`.

**Question:** Why should a hash-only verifier not claim that the
model's logits, generated text or learning algorithm are correct?
