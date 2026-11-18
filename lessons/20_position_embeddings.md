# Lesson 20: give each token its position

Token embeddings alone cannot distinguish the first occurrence of a
character from the fourth. We add a **learned absolute position vector**.

```text
token IDs [B,T] ----> token vectors [B,T,C] ------+
                                                  +--> sum [B,T,C]
position IDs [T] ---> position vectors [T,C] -----+
```

The second tensor broadcasts across the **B** batch dimension.
Learned positions are stored in a table of shape `[block_size, C]`.
This is the absolute positional-embedding approach used by GPT-2.

Run: `python examples/20_position_embeddings.py`

**Experiment:** Repeat the same token at three positions. Why do its
combined vectors generally differ?

**Checkpoint:** Why must a model reject sequences longer than its
configured position-embedding table?
