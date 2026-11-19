# Lesson 149 — Is CPU window gathering a bottleneck?

Our character-GPT sampler draws random starting positions and gathers
next-token windows of length T. To compare implementations fairly,
both must gather the **same windows from the same start positions**.

We add `gather_next_token_pairs` to separate fixed-start gathering
from the random-number draw. Tests prove that the original sampler and
the new helper produce the same token IDs for identical starts.

```bash
python examples/100_sampling_benchmark.py --batch-size 64 \
  --block-size 32 --iterations 8
```

The benchmark compares vectorized Torch indexing with a Python loop,
holding the tensor, start indices and context length fixed. It verifies
exact equality before timing either path. No particular performance
ratio is asserted because CPU threads, caches and shapes matter.

This isolates **window gathering**, not disk loading, tokenization or
full-model training.

Run `python -m pytest tests/test_fixed_window_gather.py \
tests/test_sampling_benchmark.py -q`.

**Experiment:** Increase batch size while holding context fixed.
Which implementation scales better on your CPU?

**Checkpoint:** Why would drawing new random positions within each
timed call make the comparison harder to interpret?
