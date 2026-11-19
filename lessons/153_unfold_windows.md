# Lesson 153 — Represent all overlapping windows as a view

Character-GPT samples many overlapping token contexts from one
contiguous one-dimensional token stream. Torch's `unfold` can express
the table of possible next-token windows as an overlapping **view**:

```text
tokens:   [0, 1, 2, 3, 4, 5, ...]
T = 3
window 0: [0, 1, 2, 3] → input [0,1,2], target [1,2,3]
window 1: [1, 2, 3, 4] → input [1,2,3], target [2,3,4]
...
```

The view has shape `[N-T,T+1]`, but its storage still refers to the
original `[N]` token stream. Indexing a batch of rows copies only the
selected windows, not the entire conceptual table.

We add `gather_unfold_next_token_pairs` beside the original gather
function and test that their token IDs agree exactly at both ends
of the corpus and with repeated start indices.

This commit does not alter training or assert a speedup.

Run `python -m pytest tests/test_unfold_gather.py -q`.

**Experiment:** Check the storage pointer of `tokens.unfold(0,T+1,1)`.
How does its logical shape differ from the bytes it owns?

**Checkpoint:** Why is an overlapping view less expensive than
materializing every possible window in a new tensor?
