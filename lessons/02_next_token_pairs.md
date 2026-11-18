# Lesson 02: learning to predict the next token

GPT is trained to predict the **next** token at every position, not to
memorize the token currently at that position.

```text
full sequence: 10  20  30  40
inputs (x):   10  20  30
targets (y):  20  30  40
```

Notice that `x[:, :-1]` drops the final token and `x[:, 1:]` drops the
first token. Their shapes are identical, **(B, T-1)**. The two tensors are
views over the same original batch; the data is not copied.

Run `python examples/02_next_token_pairs.py` and `python -m pytest`.

**Experiment:** Change the batch to two sequences of five tokens.
What are the two output shapes?

**Checkpoint:** Why would training fail conceptually if inputs and targets
contained the exact same tokens at corresponding positions?
