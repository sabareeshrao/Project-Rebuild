# Lesson 142 — Compare what the model learns, not just training speed

A microbatch optimization is useful only if it still trains the
**same objective**. An impressive tokens/second number is misleading
if the optimizer is learning from different examples or wrong loss
weights.

The comparison script now trains a baseline and separate streamed
runs from the same starting seed, text corpus, logical batch, model
shape, and optimizer settings:

```bash
python examples/98_compare_microbatch_learning.py \
  --steps 3 --batch-size 5 --microbatch-sizes 1,2,5 \
  --json out/microbatch_learning.json
```

Every run uses the same total optimizer updates and reports:

- Validation-loss history at identical evaluation steps.
- Final training and validation loss.
- Largest absolute difference between final model parameters and
  the unsplit baseline.
- Largest absolute difference between validation trajectories.

```text
unsplit B=5 → model parameters → validation loss by step
split M=2   → model parameters → validation loss by step
split M=1   → model parameters → validation loss by step
                       |
                compare differences
```

The comparisons are numerical, not promises of bitwise equality.
Different matrix shapes can change floating-point reduction order,
and GPU kernels may behave differently. CPU tests use a tolerance
rather than assuming identical tensor bits.

Run `python -m pytest tests/test_microbatch_learning_comparison.py -q`.

**Experiment:** Repeat the comparison with a different seed. Are the
*absolute* loss values different while the gap between strategies
remains small?

**Checkpoint:** Why is comparing final validation loss alongside
training speed a stronger test than timing microbatch updates alone?
