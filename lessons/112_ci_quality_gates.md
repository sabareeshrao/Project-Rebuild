# Lesson 112 — Treat CI as a scientific quality gate

The first test workflow introduced after commit 110 caught an actual
regression, not an attention speed problem: a changed exception message
broke an upstream compatibility test.

We now run CI in three deliberate stages:

1. **Compile** every source module, test, and example to catch syntax
   and indentation errors quickly.
2. **Test** the full CPU suite and write a machine-readable JUnit XML
   report.
3. **Archive** that report with `if: always()`, even when pytest fails.
   It remains accessible through the GitHub Actions run artifacts.

```text
push → Python setup → install → compileall → pytest
                                         ├─ pass
                                         └─ fail
                                             ↓
                                     upload JUnit report
```

CUDA-specific tests are skipped when no accelerator is present, so a
green CPU CI run is *not* evidence that the CUDA FP16, GPU SDPA and
performance paths are correct.

The benchmark scripts are intentionally **not** required to exceed a
fixed speedup. That would make CI depend on noisy shared-hardware
performance, rather than correctness.

Run locally:

```bash
python -m compileall -q src tests examples
python -m pytest -q --junitxml=pytest-results.xml
python examples/88_attention_benchmark.py --iterations 5
python examples/89_training_benchmark.py --iterations 5
```

**Experiment:** Introduce a syntax error in a scratch copy of an
example and run `compileall`; note that it fails before tests begin.

**Checkpoint:** Why should correctness tests assert numerical tolerance
but avoid asserting that SDPA must be a specific percentage faster?
