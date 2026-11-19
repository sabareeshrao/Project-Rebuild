# nanoGPT educational rebuild: 300-commit milestone review

This is an **original learning-focused implementation**, not a fork
of upstream nanoGPT's Git history. Commit counts measure the size of
the learning roadmap, not quantitative equivalence to GPT-2.

## Milestones achieved through commit 294

- Token/character data pipeline, next-token targets and baselines.
- Trainable bigram model, embeddings, attention, transformer blocks,
  stacked GPT, generation, checkpoints and resume.
- AdamW, weight decay groups, schedulers, gradient accumulation and
  CPU/GPU-oriented training infrastructure.
- Opt-in upstream nanoGPT interfaces for named tensor mapping and
  logits/loss/training comparisons with a separately trusted model.py.
- Sample coverage and validation experiments.
- Deterministic grouped-AdamW restart, fallback and fault-injection
  tests on small CPU configurations.
- Four original model-behavior checks (causal masking, batch isolation,
  forward/loss contract, finite-difference backprop).
- Four additional checks (tied parameter identity, seeded generation,
  checkpoint save/reload and manual-vs-SDPA attention).

## What constitutes actual evidence?

1. **Local unit tests** exercise tiny synthetic models and negative
   controls. Check that the applicable GitHub Actions run completed
   successfully; don't infer passing CI just from a pushed commit.
2. **Saved model conformance:** after generating the capstone checkpoint,
   the eight-check CLI writes a SHA-256-identified JSON report.
3. **Independent verification:** first check the report schema and
   source file hashes without deserializing PyTorch. Then explicitly
   rerun the checks on trusted checkpoint data.
4. **Upstream equivalence:** original nanoGPT source must be supplied
   and reviewed independently. A local adapter or renamed test double
   does not establish an upstream reproduction result.

## Reproducible evidence commands

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/120_capstone_cli.py --steps 4
python -m rebuild_gpt.extended_conformance_cli
python -m rebuild_gpt.conformance_review_cli --mode verify
python -m rebuild_gpt.conformance_review_cli --mode reproduce
python -m rebuild_gpt.conformance_review_cli --mode sweep --seed-sweep 11,29
```

Each stage checks a different claim. `verify` is read-only hashing;
`reproduce` loads **trusted** model weights and runs the checks;
`sweep` runs the checks on several deterministic held-out batches.

## Not established by this roadmap alone

- Training original GPT-2-scale weights to matching quality.
- End-to-end parity with original nanoGPT's complete distributed
  training, exact data loader, all hyperparameters and full GPU stack.
- Proof of cross-device bitwise deterministic training.
- Production-level safety, security or checkpoint authenticity.
- Generalization performance measured against published GPT-2 results.

## Finishing plan: commits 295–300

The final six commits should prioritize fixing actual CI failures,
reproducing a complete train-to-generate demonstration, collecting
opt-in actual upstream evidence where available, validating the
release documentation, and recording a final **evidence-based**
retrospective. Adjust that plan if verification discovers a bug.
