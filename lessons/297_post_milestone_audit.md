# Lesson 297 — New numerical tests on a verified saved GPT

After the original 300-commit learning milestone, we added three
checks that look for issues beyond the initial eight invariants.

This audit combines:

```text
saved trained GPT + held-out corpus + eight-check report
     |-> hash-verification of exact inputs
     |-> nine explicit numerical error budgets
     |-> three float64 numerical gradient coordinates
     |-> overlong prompt vs manual cropped-context sampling
     +-> passing JSON report (only after all checks pass)
```

Run `python -m rebuild_gpt.release_review_cli --mode train` in a
fresh output directory, followed by
`python examples/190_post_milestone_audit.py`.

The output records source hashes, measured numerical errors, the
three gradient coordinates and generated token IDs. It does not
modify the input checkpoint or existing conformance report.

**Question:** Why do cross-implementation numerical bounds and
long-context correctness still not demonstrate GPT-2-scale quality?
