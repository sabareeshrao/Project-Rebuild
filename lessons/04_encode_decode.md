# Lesson 04: encode, decode, and round-trip

**Encode** turns characters into integers. **Decode** maps those integers
back to characters. Every ID has a meaning only in its original vocabulary.

```text
text --encode--> token IDs --decode--> original text
```

Our first tokenizer is deliberately small: each Unicode character is one
token. This is **not** the byte-level BPE tokenizer used in typical GPT-2
setups, and it cannot encode a previously unseen character.

Run: `python examples/04_encode_decode.py`

**Experiment:** Build a tokenizer from `"abc"`, then try to encode `"abd"`.
Read the error and explain why it occurs.

**Checkpoint:** Why would changing the token-to-ID mapping after training
make a saved model's embeddings incorrect?
