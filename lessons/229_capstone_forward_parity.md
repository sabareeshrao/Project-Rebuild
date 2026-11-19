# Lesson 229: compare trained logits to the original nanoGPT

The old parity tests used fresh random local weights. This one
**loads the saved capstone checkpoint** and transfers those weights
to a separately supplied, reviewed upstream nanoGPT model.

On identical held-out token batches we compare:
- Full training logits, for every token position and vocabulary ID
- Masked next-token loss, with final target -1 ignored
- Final-position logits, matching the upstream inference contract

The existing strict key translator rejects mismatched tensor names
and shapes; the numerical parity test raises if output tolerances fail.

```bash
python examples/120_capstone_cli.py
python examples/122_capstone_forward_parity.py \
  --upstream-model /trusted/nanoGPT/model.py
```

**Security:** An upstream `model.py` is executable Python; review and
pin its revision yourself. Our repository never downloads or bundles it.

**Question:** Why does copying matching weights matter when measuring
whether two implementations produce the same logits?
