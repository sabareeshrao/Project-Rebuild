# Lesson 256: inspect EVERY interruption, not only the final result

Even if the final state is correct, a useful debugging record should
show how each saved restart point relates to the others.

Each checkpoint must include the same tokenizer, model shape, optimizer
controls and exact corpus identity. Its training step must match the
planned interruption. The saved evaluation history must be an
extension of the preceding segment's history.

    checkpoint at step 2 -> check identity and moments
    checkpoint at step 4 -> check identity, moments, history prefix
    checkpoint at step 6 -> check identity, moments, history prefix
                        -> compare with uninterrupted step 6

The inspector hashes each checkpoint and reports how many optimizer
state entries and evaluation records were saved.

Run: python examples/149_verify_resume_stages.py

Checkpoint: Why is the prefix relationship between evaluation
histories important when reconstructing a restarted training run?
