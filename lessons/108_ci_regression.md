# Lesson 108 — Read CI failures before optimizing further

The first GitHub Actions run at commit 110 failed **one** test. A
previous upstream-parity test expected the phrase `fused attention`
in an error for educational attention, but the new SDPA support
changed the message to `fused/SDPA attention`.

The actual restriction remains correct:
- The educational multi-head layout cannot be translated to upstream
  packed QKV weights without a conversion.
- Both manual-fused and SDPA layouts share packed QKV state keys and
  *are* supported.

We retain the informative legacy phrase in the error and add an
explicit regression test to cover both cases.

```bash
python -m pytest tests/test_upstream_parity_helpers.py \
  tests/test_nanogpt_keys.py -q
```

**Experiment:** Replace the error text with an unrelated phrase and
observe how an API contract test catches it.

**Checkpoint:** Why can a seemingly harmless message change break
a downstream program or test, even when model mathematics is unchanged?
