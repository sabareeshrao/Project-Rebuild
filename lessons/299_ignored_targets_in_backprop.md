# Lesson 299 — Ignored targets affect the gradient, not just the loss

In language-model training, target ID `-1` means **do not supervise
this position**. The model still computes its hidden state there,
but that position must contribute nothing directly to cross-entropy.

We independently compare two calculations:

```text
A: full logits + ignore_index=-1 -> loss -> backward
B: select valid target positions  -> CE   -> backward
                                       same gradients?
```

The test compares every named trainable parameter's gradient, not just
the averaged loss. It creates separate model copies and disables
dropout so their predictions use identical weights and inputs.

Run `python examples/192_masked_target_gradients.py`.

**Experiment:** Modify the model's training loss by a factor of 1.3.
Why can forward shapes still look correct while this invariant fails?
