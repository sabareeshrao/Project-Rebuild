# Lesson 234: does our GPT compute the same learning signal?

Matching predicted text alone says little about training equivalence.
Here we set the **same weights**, use the **same input and target
tokens**, compute backward passes, and compare each parameter gradient
under nanoGPT's parameter naming scheme.

We switch both models into eval mode for this experiment to disable
random dropout; gradients still flow in eval mode.

```text
shared weights + identical batch
            /           \
      our loss         upstream loss
      backward         backward
            \           /
          compare gradients
```

The actual original model is **opt-in** and must be a reviewed local
`model.py`; the tests without it use a local adapter to validate
the comparison machinery, not to claim genuine upstream parity.

```bash
python examples/127_gradient_parity.py \
  --upstream-model /trusted/nanoGPT/model.py
```

**Checkpoint:** Why can forward outputs agree within tolerance
while backward gradients still require independent checking?
