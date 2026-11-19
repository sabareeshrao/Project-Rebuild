# Lesson 222 — Compare a stratified estimate to a known exact mean

Our previous modules reweight individual GPT losses by the
number of eligible starts in their regions. We can verify the
arithmetic more transparently on a synthetic population with
a known exact uniform-start average.

Consider 21 valid starts split among four regions of sizes
[5, 5, 5, 6]. Give every start in each region the constant
score 1, 2, 3 or 4 respectively. If we sample five balanced
starts, their region counts are [2, 1, 1, 1].

```text
Raw balanced-sample mean:
  (1 + 1 + 2 + 3 + 4) / 5 = 2.2

Exact uniform-start population mean:
  (5*1 + 5*2 + 5*3 + 6*4) / 21 = 54/21 ≈ 2.5714

Region-weighted estimate: 54/21 (exact in this toy).
```

```bash
python examples/115_start_uniform_toy.py \
  --corpus-tokens 25 --context 4 --region-count 4 \
  --batch-size 1 --eval-batches 5 --seed-count 3 \
  --csv out/toy_weighting.csv --json out/toy_weighting.json
```

This exact equality is **by design**: within each toy region
all scores are constant. Real GPT losses vary with the
context start, so reweighting does not eliminate finite-sample
error or supply calibrated uncertainty intervals.

Run `python -m pytest tests/test_start_uniform_toy.py -q`.

**Checkpoint:** Why can a region-balanced unweighted mean differ
from the target population mean even though every region was sampled?
