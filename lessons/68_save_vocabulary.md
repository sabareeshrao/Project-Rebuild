# Lesson 68: preserve a tokenizer's character-to-ID mapping

A trained model's input embedding table associates each **integer ID**
with a learned vector. If a different tokenizer assigns new IDs to
the same characters, the weights no longer mean what they did during
training.

We save the **ordered vocabulary** in a versioned JSON file, then
rebuild exactly the original mapping. The loader validates that every
entry is a single character and that the list is sorted and unique.

```text
training text -> CharTokenizer -> ordered vocab -> vocab.json
vocab.json -> same ordered vocab -> same token IDs -> GPT
```

Run: `python examples/65_save_vocabulary.py`.

**Experiment:** Reverse the stored vocabulary. Why does the loader
reject it instead of silently assigning new IDs?

**Checkpoint:** Why is it insufficient to save the model weights
without also saving the tokenizer that produced its training IDs?
