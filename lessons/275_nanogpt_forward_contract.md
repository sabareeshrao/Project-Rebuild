# Lesson 275: training and inference do not return identical shapes

Original nanoGPT intentionally exposes two related forward contracts.

**Training:** input IDs and next-token targets have shape [B,T].
The model produces logits [B,T,V] and one averaged cross-entropy
loss over all targets **except -1**.

**Inference:** with no targets, the model returns only the logits
for the final time position, shaped [B,1,V], and no loss.

```text
[B,T] tokens + targets -> [B,T,V] logits + masked CE
[B,T] tokens           -> [B,1,V] last logits + None
```

The test independently recomputes masked cross-entropy from the
unoptimized forward logits, so it catches incorrect masks, shapes
and outputs even if the model's prediction accuracy appears good.

Run `python examples/168_forward_contract.py`.

**Checkpoint:** Why is target ID `-1` not a new vocabulary symbol?
What would happen if you treated it as a real prediction class?
