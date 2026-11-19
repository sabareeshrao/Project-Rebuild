# Lesson 230: identical weights, identical sampled tokens?

Matching the logits is an important prerequisite. Here we go one step
further: compare the *sampled token IDs* using the same weights, prompt,
top-k setting, temperature and random seed in both implementations.

```text
saved model + vocabulary -> prompt IDs
                              |
                        local GPT / upstream GPT
                              |
                        sample each with same seed
                              |
                       token IDs exactly equal?
```

We reuse the existing `compare_generation_against_upstream` helper
so that both models are placed in eval mode and the caller's RNG state
is restored. Exact sampling can be more sensitive than close logits.

After creating `out/capstone/model.pt`, run:
```bash
python examples/123_capstone_generation_parity.py \
  --upstream-model /trusted/nanoGPT/model.py --prompt a
```

Only supply source you have reviewed: importing upstream `model.py`
executes Python code. No source is fetched automatically.

**Question:** Why might two nearly identical probability
distributions still produce different samples?
