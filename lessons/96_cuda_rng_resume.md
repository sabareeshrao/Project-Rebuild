# Lesson 96 — GPU dropout needs a GPU RNG checkpoint

A saved PyTorch CPU RNG state is not enough to reproduce a CUDA run.
Dropout and CUDA kernels can draw from CUDA random streams. If we
restart a model with the correct weights and optimizer, but a different
GPU RNG stream, the next training update may change.

This checkpoint now retains two distinct sources of randomness:

- `torch.get_rng_state()` — CPU RNG, used for CPU operations
- `torch.cuda.get_rng_state_all()` — states of available CUDA devices

And a third, local `torch.Generator` used to sample token windows.

A resumed **CUDA** run restores all CUDA states after constructing the
model and optimizer. CPU-only checkpoints still load without CUDA.

```text
weights + AdamW state + CPU RNG + CPU batch generator + CUDA RNG(s)
                                |
                        restore and continue
```

Run: `python -m pytest tests/test_text_resume.py -q`

**Experiment:** Train a model with dropout on CUDA, save, then resume.
Why can the next forward pass differ without saved CUDA RNG state?

**Checkpoint:** Why is it important to restore the RNG state *after*
rebuilding the model (whose initialization itself draws random numbers)?

This does not yet support multi-process distributed training or mixed-
precision gradient scaler state.
