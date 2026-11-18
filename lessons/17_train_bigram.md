# Lesson 17: from text to a trainable text generator

This commit connects everything we have built since Lesson 01.

```text
tiny_story.txt -> tokenizer -> token stream
                                  |
                           train / validation
                                  |
                           sample [B,T] batches
                                  |
                         bigram embedding [V,V]
                                  |
                         predicted logits [B,T,V]
                                  |
                         cross-entropy loss
                                  |
                         backward + AdamW step
                                  |
                     generated tokens -> decode
```

The model now has **trainable parameters**. Unlike our uniform and
unigram baselines, the bigram table learns a different next-token
distribution for each current token.

The example samples a single seeded validation batch *before* training
and evaluates the same batch afterward. This isolates changes in model
weights from changes in sampled validation examples. On such a small
corpus, validation loss may vary and need not improve.

Run: `python examples/17_train_bigram.py`

**Experiment:** Try 10 versus 120 training steps. What happens to
training loss, held-out loss, and generated characters?

**Checkpoint:** Why can our trained bigram generator still make
nonsensical text even when its loss decreases?

**Next:** Study vector embeddings, positional information, and why
self-attention can use more than the latest token.
