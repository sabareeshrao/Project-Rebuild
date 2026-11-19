# Lesson 133 — Split a logical batch into microbatches

One optimizer update can use **B examples** while processing only
**M examples at once**. Dividing B=10 into microbatches of M=4 gives:

```text
logical batch: [ 0 1 2 3 4 5 6 7 8 9 ]
                └──4──┘ └──4──┘ └2┘
microbatches:      4      4      2
```

The final microbatch need not be full. We must not discard its
examples or repeat them: the original example order is preserved.
A slice is a PyTorch view, so splitting does not itself copy the
entire logical batch.

The `microbatch_sizes` planner accepts only positive integer sizes.
`split_microbatches` accepts targets of shape `[B]` for a
classifier or `[B,T]` for our causal language model.

This commit implements only splitting. Correctly weighting losses
from differently sized microbatches comes next.

Run `python -m pytest tests/test_microbatches.py -q`.

**Experiment:** Split B=13 with M=5. Predict the chunk sizes.

**Checkpoint:** Why must the last, smaller microbatch be accounted
for rather than averaging every microbatch loss equally?
