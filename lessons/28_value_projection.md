# Lesson 28: values carry the information

Our last five commits computed **attention weights** using Q and K.
But attention weights only tell us *where to look*. We still need the
information that will be brought together: the **value vectors (V)**.

```text
hidden [B,T,C] -> learned W_V [C,D] -> values [B,T,D]
```

Like query and key projections, a value projection is a learnable
linear transformation (no bias in our small first head).

Run: `python examples/28_value_projection.py`

**Experiment:** Change the value head width from 3 to 6. What changes
about the value tensor and number of parameters?

**Checkpoint:** What is the difference between a key (used to
calculate *relevance*) and a value (used to provide *content*)?
