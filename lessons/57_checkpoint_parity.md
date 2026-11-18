# Lesson 57: verify predictions survive both checkpoint paths

A useful compatibility test checks **behavior**, not merely that
a file can be opened. We now run this end-to-end experiment:

```text
Train one GPT step
   -> predictions A
   -> save native checkpoint
   -> load native checkpoint -> predictions B == A
   -> export nanoGPT-style tensor names
   -> import them into another model -> predictions C == A
```

Both the educational and fused paths also have native checkpoint
round-trip coverage. The fused path is required for nanoGPT-style
tensor keys. We verify that tied weights remain tied throughout.

Run: `python examples/57_checkpoint_parity.py`.
Then run `python -m pytest` to check the complete project.

**Experiment:** Turn dropout on and forget to call `eval()`.
Why might two successive forward passes differ even without
changing the weights?

**Checkpoint:** What's the difference between confirming that
*our own* exported tensors round-trip and actually loading an
**original upstream** checkpoint with its own configuration?

**Next:** Add comparisons against actual original nanoGPT model
weights, configuration conventions and generation behavior. Do not
treat this step as proof of full upstream checkpoint compatibility.
