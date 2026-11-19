# Lesson 223: write down the experiment before training

A repeatable experiment starts with its inputs. Our `CapstoneSpec` records
the model size, optimizer settings, random seed, and sampling settings.
The first capstone is deliberately tiny and runs on **CPU**.

```text
corpus + settings -> training -> checkpoint -> text sample
```

The seed controls pseudo-random choices on a given implementation, but
does **not** guarantee bit-for-bit identical results across hardware or
PyTorch versions. Training and sampling settings are separate: a prompt
is not an argument to the optimizer.

Run: `python examples/116_capstone_settings.py`

**Experiment:** Change `n_embd` from 8 to 12 and `n_head` from 2 to 3.
Why is 11 embedding channels with 3 heads rejected?

**Checkpoint:** Which settings describe the model structure, which
control training, and which only affect text generation?
