# Lesson 03: create a vocabulary

A language model consumes integer IDs, not raw strings. Before we can
encode text, each unique character needs a stable ID.

We start with **character-level** tokenization to make the mapping visible.
Later we will study the subword/tokenization approach used by larger GPTs.

For the text `cab cab`, sorting the set of characters produces:

```text
" " -> 0
"a" -> 1
"b" -> 2
"c" -> 3
```

Sorting makes the mapping independent of the order in which characters appear.
The vocabulary size is 4, even though the text has 7 characters.

Run: `python examples/03_character_vocabulary.py`

**Experiment:** Reorder the characters but keep the same unique symbols.
Does the vocabulary change?

**Checkpoint:** Why must token IDs be consistent during training and generation?
