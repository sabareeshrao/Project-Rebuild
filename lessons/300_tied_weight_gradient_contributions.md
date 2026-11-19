# Lesson 300 — A single tied embedding has two gradient paths

In nanoGPT, the input token embedding and the final output projection
are the **same trainable parameter**. The shared weight participates
twice in the computation graph, so backward accumulates two sources.

```text
                     shared embedding weight
                     /                    \
           token lookup                 output projection
                     \                    /
                      next-token loss
                            |
               gradient = input + output
```

To prove this rather than assume it, we clone the GPT twice. One copy
retains the shared Parameter. On the other, we split the head's weight
into a separate Parameter with identical initial values.

Their forward predictions match. After backward, the tied parameter's
gradient must equal the **sum** of the two gradients in the untied
copy, and all other parameter gradients must still agree.

Run `python examples/193_tied_gradient_contributions.py`.

**Question:** Why would copying the embedding into an independent
head Parameter silently change training despite initially identical
predictions?
