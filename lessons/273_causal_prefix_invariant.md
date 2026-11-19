# Lesson 273: a GPT cannot peek into the future

The defining property of a **causal** language model is that the
prediction at position *t* depends only on tokens through *t*.

We can test this without knowing the correct predictions in advance:

```text
tokens:    A B C D       A B C X
logits:    p p p q       p p p r
           ^^^^^         ^^^^^
           earlier positions MUST match
```

Change only the final token, then compare all logits at earlier
positions. A proper causal mask prevents any influence from that
future token. The model is temporarily put in eval mode so dropout
does not falsely look like leakage.

Run: `python examples/166_causal_invariance.py`

The tests also include a deliberately broken model that broadcasts
the final token into every prediction. The invariant catches this.

**Question:** Why might a language model achieve a low training loss
while still being invalid for left-to-right text generation?
