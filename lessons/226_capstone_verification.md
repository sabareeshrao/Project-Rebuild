# Lesson 226: verify a result independently of training

Re-running the optimizer is not necessary to check a saved model's
**inference**. We verify the input corpus hash, checkpoint hash and
saved sample, then reload the checkpoint and replay the seeded sampler.

```text
run.json + corpus + model.pt + sample.txt
              |
        SHA-256 comparisons
              |
       reload and regenerate
              |
        identical text?
```

Run `examples/118_capstone_bundle.py` first, then run
`python examples/119_capstone_verify.py`.

**Experiment:** Edit `out/capstone/sample.txt`, rerun verification
and examine the error. Restore it by rerunning the bundle example.

**Checkpoint:** Explain why matching hashes and replayed text
establishes artifact consistency but does **not** prove that a model
trained correctly, generalized to new text, or will behave identically
on every CPU/PyTorch version. Only load checkpoints you trust.
