# Lesson 224: connect the recipe to a real optimizer

We call the existing `train_char_gpt` function instead of maintaining
a second, almost identical optimizer and training loop.

The flow is still real:
```text
UTF-8 corpus -> tokens -> sampled batches -> GPT -> cross-entropy
                            ^                     |
                            +---- AdamW gradients-+
```

Run: `python examples/117_capstone_train.py`

Three updates on a tiny corpus are **not** evidence of a useful language
model. The held-out loss may even rise. What matters is the ability to
trace data, parameters, loss, and optimizer updates with a repeatable seed.

**Experiment:** Compare runs of 1 step and 3 steps. Do they always
improve validation loss on this tiny sample?

**Checkpoint:** Why reuse an already-tested training loop instead of
copying it into the capstone?
