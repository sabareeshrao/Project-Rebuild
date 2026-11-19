# 300 planned educational commits — evidence-based milestone

The learning roadmap reached its 300th numbered commit with lesson 293.
It is an original **character-level GPT rebuild inspired by nanoGPT**,
not a claim to reproduce the original GPT-2 training results.

## Demonstration

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m rebuild_gpt.release_review_cli --mode train \
  --corpus data/tiny_story.txt --output-dir out/release_review
python -m rebuild_gpt.release_review_cli --mode verify
python -m rebuild_gpt.release_review_cli --mode reproduce
python examples/186_release_evidence.py
```

The last three commands separate hash-only validation, independent
behavioral rechecking, and consistency of the complete evidence chain.

## What we built

A tiny trainable GPT; character tokenizer; causal attention, embeddings,
transformer blocks, generation and checkpoint loading; optimizer and
scheduler controls; CPU training resumption and recoverability tests;
opt-in upstream tensor, forward and generation parity adapters;
and tests for eight independent model behaviors.

## Verification boundary

- Check the **completed** GitHub Actions run for the exact commit SHA.
  Pushing code does not establish passing CI.
- A successful CPU demonstration establishes only the documented
  small-model results; it does not prove GPT-2-quality text or full
  equivalence with original nanoGPT's distributed training.
- Actual upstream comparisons require a **trusted, reviewed**
  source `model.py`. No external model source runs automatically.
- The evidence files are SHA-256-identified but are not signed.

Subsequent numbered lessons may extend the educational work beyond
this milestone. See `docs/RELEASE_PREFLIGHT.md` for the release workflow.

## Extensions after commit 300

Commits 301–304 continue learning with explicit numerical error
budgets, multi-coordinate gradient checks, long-prompt context
validation, and an opt-in post-milestone audit. They extend the
original educational plan; they do not change the historical
300th-commit milestone or imply upstream GPT-2 parity.
