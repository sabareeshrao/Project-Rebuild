# Lesson 75: remember the model that validated best

The **last** training step need not produce the **best**
held-out result. A model can begin to overfit, and validation
loss may rise even as training loss falls.

`BestValidationModel` remembers the parameters from the lowest
observed validation estimate. It clones **every state-dict tensor**,
so later optimizer updates cannot silently change the saved snapshot.

```text
step 0   validation=1.1  -> save snapshot
step 10  validation=0.8  -> replace snapshot
step 20  validation=1.0  -> retain step 10
after training           -> optionally restore step 10
```

Ties are only replaced by **strict improvements**, not equal scores.
Our tests also check that restoring a GPT preserves its tied token
embedding/output weight parameter.

Run: `python examples/72_best_model_snapshot.py`.

**Experiment:** Give the tracker decreasing and then increasing
validation scores. At which step will it restore weights?

**Checkpoint:** Why is storing `model.state_dict()` without
cloning tensor values unsafe when the optimizer continues updating?

This tracker doesn't decide how often to evaluate or where to save
a file yet. Those are the next commit's responsibilities.
