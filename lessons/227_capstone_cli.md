# Lesson 227: a reproducible experiment you can run end to end

The final capstone command connects every layer without hiding its steps:

```text
corpus -> tokenizer -> token batches -> stacked GPT
       -> optimizer updates -> validation losses
       -> model.pt + vocabulary -> seeded sample
       -> run.json with inputs, metrics, and SHA-256 digests
       -> independent verification without training
```

From the repo root (after `pip install -e ".[dev]"`):

```bash
python examples/120_capstone_cli.py --steps 4
python examples/120_capstone_cli.py --verify-only
```

Outputs live in `out/capstone` and are excluded from Git. To keep
different runs, set `--output-dir` to a different folder.

**Experiment:** Use a second output directory and change only the seed.
Compare text, metrics and corpus hashes. The corpus hash stays the same;
weights and sampled text can change.

**Checkpoint:** Explain which outputs can be verified without training,
and why the same training settings do not imply identical results on
different hardware/PyTorch versions.

The five capstone commits do **not** replace the full nanoGPT training
pipeline or demonstrate GPT-2 reproduction. They demonstrate an honest,
auditable miniature end-to-end system built from our earlier lessons.
