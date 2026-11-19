# Lesson 292 — Release evidence, not release hype

Our last five lessons connect training to generation to conformance:

```text
train tiny GPT -> model.pt + sample.txt + run.json
      |-> replay generated sample
      |-> hash-check original corpus and checkpoint
      |-> run eight model checks
      |-> verify both reports describe the same weights
      +-> rerun and compare the eight measurements
```

A single CLI gives you three modes: **train** (new run), **verify**
(read-only hash checking), and **reproduce** (load trusted model,
re-run checks, and save a reproduction audit).

Run:
```bash
python examples/185_release_review.py --output-dir out/release_review
python -m rebuild_gpt.release_review_cli --mode verify
python -m rebuild_gpt.release_review_cli --mode reproduce
```

**Question:** Which statements about GPU scaling, GPT-2 language
quality or complete upstream nanoGPT parity would remain
unjustified even after all these commands pass?

Read `docs/RELEASE_PREFLIGHT.md` before the final commit.
