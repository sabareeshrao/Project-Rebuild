# Lesson 287 — One command, three levels of evidence

The same saved GPT can be reviewed at different costs and levels
of confidence. The new CLI explicitly separates:

1. **verify** — validate report schema and hash model/corpus bytes;
   no PyTorch checkpoint deserialization.
2. **reproduce** — load the trusted checkpoint, rerun all eight checks
   and compare with the previous report's recorded measurements.
3. **sweep** — draw new deterministic held-out token windows for
   multiple seeds; require all eight checks to pass each time.

```bash
python examples/120_capstone_cli.py --steps 4
python -m rebuild_gpt.extended_conformance_cli
python examples/180_conformance_review.py --mode verify
python examples/180_conformance_review.py --mode reproduce
python examples/180_conformance_review.py --mode sweep --seed-sweep 11,29
```

If a command fails, diagnose the actual mismatch; don't replace it
with a manually edited `"passed"` JSON report.

**Question:** Why does passing eight local model invariants still
fall short of demonstrating GPT-2-scale training quality?

See `docs/MILESTONE_AUDIT.md` for the scope and limitations of the
complete 300-commit educational milestone.
