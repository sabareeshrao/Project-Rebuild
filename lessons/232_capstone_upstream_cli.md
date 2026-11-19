# Lesson 232: compare a real saved checkpoint with nanoGPT

This capstone milestone joins the previous four lessons into one
command. It is **opt-in** because importing an original `model.py`
executes arbitrary Python from that file. You must supply a
locally reviewed, trusted source and ideally a SHA-256 pin.

The command:
1. Verifies the upstream source checksum before importing it.
2. Loads our saved model and vocabulary.
3. Samples identical held-out input/target windows.
4. Checks matched weights, logits and cross-entropy loss.
5. Checks exact generation token IDs at the same seed.
6. Writes a passed JSON report only if both comparisons succeed.

```bash
python examples/120_capstone_cli.py --steps 4
python examples/125_capstone_upstream_cli.py \
  --upstream-model /trusted/nanoGPT/model.py \
  --upstream-sha256 <real-source-sha256>
```

The optional genuine-reference regression test is skipped unless
`NANOGPT_MODEL_PY` points to an actual reviewed file. Mocked CI tests
check that commands pass arguments and reject failures; they do **not**
by themselves establish parity with Karpathy's implementation.

**Experiment:** Pass an intentionally incorrect digest and verify that
the source is rejected before Python code is executed.

**Checkpoint:** Distinguish matching saved-model inference, matching
an entire optimization trajectory, and reproducing full GPT-2 training.
