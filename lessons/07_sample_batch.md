# Lesson 07: sample context windows for GPT training

Our dataset is one long stream of token IDs. A model optimizer needs a
mini-batch: multiple short contexts and their corresponding targets.

For block size 3, a possible training example is:

```text
stream:  10 20 30 40 50 60 ...
x:         20 30 40
y:         30 40 50
```

A window contains **block_size + 1** consecutive tokens. Removing its
last/first token gives the input/target pair. We draw independent random
start positions, so windows can overlap.

`sample_batch(stream, batch_size=B, block_size=T)` returns two integer
tensors, each shaped **(B, T)**.

Run: `python examples/07_sample_batch.py`

**Experiment:** Fix the random seed and run twice. Then change the seed.
What changes? What stays the same?

**Checkpoint:** If the stream has N tokens and the context has T tokens,
how many distinct valid starting positions are there? (Answer: N - T.)

**Next:** Feed these batches into an embedding table and begin computing
the model's token predictions.
