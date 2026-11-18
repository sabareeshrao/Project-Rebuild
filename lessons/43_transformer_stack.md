# Lesson 43: stack transformer blocks to increase depth

One decoder block transforms `[B,T,C]` into the **same shape**.
That allows another block to process its output. With three blocks:

```text
embeddings [B,T,C]
   -> Block 1 [B,T,C]
   -> Block 2 [B,T,C]
   -> Block 3 [B,T,C]
```

Every block has **independent parameters**. Using an `nn.ModuleList`
registers them for optimization and checkpoint storage. We do not
reuse the same block object repeatedly.

Stacking causal operations preserves causality: a position can only
receive information from itself and earlier positions, regardless
of the number of layers. Our test checks both future-token changes
and full-sequence versus shorter-prefix results.

Run: `python examples/43_transformer_stack.py`.

**Experiment:** Change `n_layer` from 3 to 1 and compare the
parameter count and returned tensor shape.

**Checkpoint:** Why does each block retain the same hidden width,
even though the network gets deeper?
