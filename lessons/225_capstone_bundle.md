# Lesson 225: make a training result usable later

Training gives us model weights in memory. A real experiment also needs
a durable artifact that we can reload and inspect independently.

```text
train -> save model.pt -> reload weights + vocabulary -> sample text
                          |
                          +-> run.json (settings, hashes, observed metrics)
                              sample.txt (generated output)
```

The checkpoint stores both weights and the **character vocabulary**.
The manifest records the SHA-256 hashes of the original corpus bytes
and saved checkpoint bytes so that later verification can detect changes.

Run: `python examples/118_capstone_bundle.py`

**Experiment:** Open the JSON manifest. Which fields describe inputs,
and which are measured outputs? Why is the hash not a quality metric?

**Checkpoint:** Why must we save the token-to-ID mapping with model weights?
