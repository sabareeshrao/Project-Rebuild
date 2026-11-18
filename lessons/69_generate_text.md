# Lesson 69: connect the GPT generator to real characters

We now have the complete inference path:

```text
"a tiny" -> CharTokenizer.encode -> [token IDs]
          -> GPT.generate -> [prompt IDs + new IDs]
          -> CharTokenizer.decode -> generated text
```

The tokenizer's vocabulary size must **match** the GPT configuration,
or the same integer ID could have a different meaning. An unseen
prompt character raises an explicit error, instead of being silently
mapped to a different token.

`generate_text()` temporarily calls `model.eval()` to disable
dropout and restores the original mode on return or failure.

Run: `python examples/66_generate_text.py`.

**Experiment:** Change the sampling seed, temperature, and top-k,
while keeping the model weights fixed.

**Checkpoint:** Why is this generated text mostly random until we
actually train the GPT on a text corpus?
