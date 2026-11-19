# Lesson 307 — Four loss checks on a saved tiny GPT

Understanding cross-entropy requires more than printing one loss
number. The new audit uses a trusted saved GPT and a deterministic
held-out corpus batch to verify:

```text
saved checkpoint + corpus
    |-> logit shift: same softmax, masked loss and logit gradients
    |-> perplexity: compare CE against uniform vocabulary baseline
    |-> per-token NLL: reconstruct loss only from supervised positions
    +-> masked microbatches: weighted sum agrees in loss and gradient
                        |
                    passed JSON
```

The first sampled sequence has only one valid target, while the other
sequences retain their remaining supervised tokens. This makes the
microbatch experiment genuinely **uneven**.

Run `python -m rebuild_gpt.release_review_cli --mode train` into
a new directory if needed, then
`python examples/200_next_token_loss_audit.py`.

The report records source hashes, measured values and token counts,
but makes no claims about GPT-2 scale or general language quality.

**Question:** Which of the four checks would catch a trainer that
incorrectly averages per-microbatch losses instead of weighting them
by valid target counts?
