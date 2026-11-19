# Lesson 221 — Compare saved checkpoints using stratified window losses

The new `examples/114_weighted_checkpoint_comparison.py` loads
two self-contained GPT inference checkpoints, verifies identical
character-to-ID mappings, and samples the same balanced validation
starts for both.

```bash
python examples/114_weighted_checkpoint_comparison.py \
  --baseline out/before.pt --candidate out/after.pt \
  --corpus data/tiny_story.txt --sampling-regions 4 \
  --batch-size 4 --eval-batches 3 \
  --csv out/weighted_paired.csv --json out/weighted_paired.json
```

CSV includes raw and population-weighted paired differences
along with baseline/candidate losses. JSON preserves the
per-seed region counts, eligible-start population sizes,
region weights, and mean paired differences by region.

This *does not update model weights* or alter the training CLI.
The reweighted estimate targets a uniformly selected eligible
start when draws within each region are uniform.

Even an unbiased estimator of a finite corpus's uniform-start
loss is **not** a guarantee of generalization or a calibrated
confidence interval. Sampled windows overlap and share text.

We also correct a prior test whose supposed invalid 30-region
case was actually valid for a corpus with 57 eligible starts.

Run `python -m pytest tests/test_weighted_checkpoint_comparison_cli.py \
tests/test_sampling_policy_audit.py -q`.

**Checkpoint:** Why is it important to preserve both raw and
population-weighted deltas in a checkpoint evaluation report?
