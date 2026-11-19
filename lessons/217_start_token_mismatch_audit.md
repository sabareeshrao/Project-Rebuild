# Lesson 217 — Detect balanced starts that still leave token gaps

An evaluator can have a **perfectly balanced** regional start
histogram while leaving whole regions of held-out text untouched.

Start regions partition valid *window starts*. Token-position
bins partition *corpus tokens*. A context window touches T+1
token positions, can cross a regional boundary, and can repeat
positions even when starts are evenly distributed.

`audit_start_and_token_coverage` evaluates both views on the
same fixed index plans, checks that all counts reconcile with
the global token union, and flags a particular mismatch:

```text
All seeds touch every start region?        yes
Any token-position bin never visited?      yes
Start balance hides token gaps?            yes
```

The flag is a **diagnostic observation**, not a statistical
significance test or an estimator-bias proof. Zero empty
token bins does not mean independent held-out observations or
complete coverage of every bin.

The model-free balanced-region sweep now includes the mismatch
flag in its CSV/JSON report:

```bash
python examples/113_balanced_region_sweep.py \
  --corpus data/tiny_story.txt --region-counts 2,4,8 \
  --position-bins 8 --json out/balance_audit.json
```

Run `python -m pytest tests/test_sampling_policy_audit.py \
tests/test_balanced_region_sweep.py -q`.

**Experiment:** Change `position-bins` from 4 to 16 without
changing the sampled index plans. Can new gaps become visible?

**Checkpoint:** Why should a model comparison report both
start-index balance and token-position coverage rather than
interpreting either measurement alone?
