# Lesson 72: train once, generate text in another process

The project now supports a complete local text workflow:

```text
text corpus -> CharTokenizer -> token IDs -> train GPT
           -> save (configuration + weights + vocab)
           -> reload for inference -> encode prompt
           -> autoregressive generate -> decode characters
```

Run the all-in-one example:

```bash
python examples/69_text_to_text.py
```

Or save a reusable inference checkpoint, then generate using a CLI:

```bash
python examples/68_train_text_gpt.py --steps 60 --output tiny_story_gpt.pt
python -m rebuild_gpt.text_cli \
  --checkpoint tiny_story_gpt.pt --prompt "a" \
  --new-tokens 80 --temperature 0.8 --top-k 5 --seed 13
```

Use only characters present in the saved training vocabulary. If
the prompt contains an unseen character, the CLI reports an error
instead of changing the mapping or guessing.

The saved inference bundle is **not** a full optimizer checkpoint.
Use the earlier training-checkpoint lesson if your goal is to
resume optimization rather than generate text.

**Experiment:** Save the model, close the training process, and run
the CLI twice using the same seed. Are the outputs identical?

**Checkpoint:** Why must a production text generator retain tokenizer
metadata, even if its model weights and architecture are available?

**Next:** More realistic training data, evaluation, optimizer
improvements, and tokenization closer to original nanoGPT.
