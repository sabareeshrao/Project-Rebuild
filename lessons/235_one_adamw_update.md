# Lesson 235: is one training update equivalent?

After verifying gradients, take **one** optimizer step for each model.
Both receive the same starting weights, same batch, same computed
gradients and same explicit `torch.optim.AdamW` hyperparameters.

```text
matching gradients -> AdamW step (local / original)
                              |
                      updated parameters
                              |
                       compare all keys
```

This exercise operates on in-memory model objects; the saved capstone
checkpoint stays unchanged. The example intentionally does **not**
invoke the original nanoGPT `configure_optimizers` with its decay
parameter groups. That is a different comparison and a later lesson.

```bash
python examples/128_one_step_adamw_parity.py \
  --upstream-model /trusted/nanoGPT/model.py
```

**Experiment:** Repeat with different learning rates. Should both
implementations remain in agreement? Does the size of the parameter
change grow as the learning rate grows?

**Checkpoint:** Why is matching one AdamW update insufficient to
claim equivalent full training trajectories?
