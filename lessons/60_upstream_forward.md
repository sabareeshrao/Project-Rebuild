# Lesson 60: upstream nanoGPT forward semantics

Our educational `GPTLanguageModel.forward(inputs)` has always
returned **all** next-token logits `[B,T,V]`. That is useful
for inspecting every prediction.

Original nanoGPT's `GPT.forward(idx, targets=None)` instead returns
a tuple `(logits, loss)`:

- With no targets, logits contain only the **last** position,
  shaped `[B,1,V]`, and loss is `None`.
- With targets, logits cover **every position**, and loss is
  cross-entropy with target ID `-1` ignored.

We add `model.forward_nanogpt(inputs, targets=None)` rather than
breaking the earlier `forward` and `loss` lessons. Its output
contract can be tested against the real upstream implementation.

Run `python examples/58_upstream_forward.py`.

**Experiment:** Mask one target with `-1`. How does the training
loss compare to a calculation that includes every target?

**Checkpoint:** Why does the original implementation save inference
work by computing vocabulary logits for only one position?

Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
