# Lesson 05: from a text file to a token stream

A language model trains on data: a long sequence of integer tokens. Our
original tiny story is checked into `data/tiny_story.txt` so you can inspect
every character, including spaces and newline characters.

```text
UTF-8 text file → vocabulary → encoded text → 1-D torch.long tensor
```

The tensor's shape is `[number_of_tokens]`, **not** `[batch, time]` yet.
We will sample small context windows from it in a later commit.

Run: `python examples/05_load_corpus.py`

**Experiment:** Add an exclamation point to the story, reload, and note how
the vocabulary size (and some token IDs) change.

**Checkpoint:** Why should we save the tokenizer's vocabulary alongside
a trained model's weights?
