# Lesson 89: a best model is not necessarily the latest model

Early stopping and model selection use the **minimum validation loss**.
When training is interrupted, keeping only current weights would lose the
historically best snapshot. Its loss, update step and independent tensors
must travel with the optimizer state.

```text
current weights → optimizer updates → next training checkpoint
best weights    → recorded on improvement → optional final model
```

The new `BestValidationModel.state_dict()` and `load_state_dict()`
store and restore this independent snapshot. We clone tensors so
mutating a checkpoint object cannot silently alter the tracker.

Run: `python -m pytest tests/test_best_resume.py -q`

**Experiment:** Change a model's weights after its best state was recorded;
then restore. Verify you get the earlier weights.

**Checkpoint:** Why must the resumable training checkpoint be written
**before** replacing the current model with its best validation weights?
