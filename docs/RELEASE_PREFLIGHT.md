# Release preflight — educational nanoGPT rebuild

The project has 299 planned learning commits after this phase.
**A commit count is not a correctness certificate.** This checklist
distinguishes actual executable evidence from tasks still awaiting
external verification.

## Run the local tests before claiming a clean milestone

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src tests examples
python -m pytest -q
```

Review GitHub Actions for the exact final commit SHA and confirm a
**completed** successful run. A workflow that is queued or running
is not evidence of passed CI.

## Produce a new train-to-evidence experiment

From the repository root, use a new output directory:

```bash
python -m rebuild_gpt.release_review_cli \
  --mode train --corpus data/tiny_story.txt \
  --output-dir out/release_review
```

This really trains a small character GPT on CPU. It writes the
model checkpoint, one seeded generated sample, the training manifest
and the eight-check conformance report.

```bash
# No torch.load, training or generation: verify hashes and report IDs.
python -m rebuild_gpt.release_review_cli --mode verify

# Load a trusted checkpoint and recompute all eight measurements.
python -m rebuild_gpt.release_review_cli --mode reproduce
```

The final command creates `out/release_review/reproduction.json`
only if the measurements reproduce and both reports identify the
same checkpoint and corpus. Older success reports are cleared before
rerunning this check.

## What a passing experiment does establish

- A tiny GPT was trained, saved and sampled using reproducible seeds.
- Its saved inference checkpoint can be loaded, its sample replayed,
  and the exact artifact bytes identified.
- The eight specified local conformance checks passed on held-out
  tokens, including gradient and attention-backend comparisons.
- Those recorded measurements can be independently rerun on the
  identical checkpoint and text corpus.

## What a passing experiment does NOT establish

- Matching GPT-2-scale training loss, benchmark scores or model quality.
- Real parity with all of upstream nanoGPT's training script.
- GPU or multi-machine determinism, large-scale training throughput,
  checkpoint authenticity, or safety for untrusted model files.
- That the original nanoGPT `model.py` was executed: genuine
  upstream comparison is an *opt-in* experiment needing reviewed
  source and an exact SHA-256 pin.

## Final commit 300

Once CI is green and the preflight has been inspected, the final
commit should record an evidence-based retrospective and close or
explicitly document any outstanding failures. Never invent a
passing upstream comparison or a performance result.
