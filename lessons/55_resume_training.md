# Lesson 55: resume training, not just the model's predictions

Saving model weights lets us run inference, but does **not** restore
the optimizer's momentum statistics or learning rate. To resume
training faithfully, also store:

```text
model config + model state_dict + AdamW state_dict
+ optimizer step number + CPU RNG state
+ (optional) batch sampling generator state
```

We reconstruct the GPT and AdamW optimizer, load their states, and
optionally restore the process-wide CPU RNG **after** reconstruction.
This matters because initializing new model parameters consumes random
numbers. It also matters when dropout is active.

Run: `python examples/55_resume_training.py`.

**Experiment:** Modify the checkpoint to omit AdamW's state and try
resuming. Why might the next optimizer update differ?

**Checkpoint:** Why is `restore_rng=True` optional rather than
silently changing the global random state for every caller?

**Scope:** This lesson supports single-process CPU RNG and standard
AdamW. GPU/distributed RNG states and complex optimizer parameter-group
layouts are not yet covered. Only load trusted checkpoint files.
