# Lesson 306 — An unweighted mean of microbatch losses can be wrong

Suppose the full batch has four supervised next-token targets, but
the first microbatch has just one and the second has three.

```text
microbatch A: 1 valid token   -> mean loss A
microbatch B: 3 valid tokens  -> mean loss B

correct combined loss = (1*A + 3*B) / 4
incorrect mean        = (A + B) / 2
```

The rule applies to both the **scalar loss and its gradient**. We
independently compare a full-batch masked cross-entropy calculation
against two smaller slices, weighted by supervised target counts.

The tests also include a fully masked microbatch: it must contribute
zero weight, not a NaN from an all-ignored cross-entropy reduction.

Run `python examples/199_masked_microbatches.py`.

**Checkpoint:** How could a training loop silently bias gradient
accumulation if it averages microbatch losses despite uneven masks?
