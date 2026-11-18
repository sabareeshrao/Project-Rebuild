# Lesson 15: one gradient-based learning step

Computing gradients is not enough: we must use them to change weights.

```text
1. model.train()                     # select training mode
2. optimizer.zero_grad(set_to_none=True)
3. loss = model.loss(inputs, targets)
4. loss.backward()                   # differentiate
5. optimizer.step()                  # update parameters
```

We use **AdamW**, a gradient-based optimizer. A small toy sequence
alternates between tokens 0 and 1; the bigram model can learn
`0 → 1` and `1 → 0`.

Run: `python examples/15_train_step.py`

**Experiment:** Remove `optimizer.step()`. How does loss change over
repeated passes? Restore it, then try different learning rates.

**Checkpoint:** Why must gradients be reset before the next update?
