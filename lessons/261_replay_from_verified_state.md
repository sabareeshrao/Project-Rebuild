# Lesson 261: turn an audited checkpoint back into a training run

After locating the latest intact intermediate checkpoint, we can
continue training to the original target step.

The replay uses the existing `train_char_gpt` implementation and
exactly the same optimizer settings and learning-rate schedule as
the initial multi-interruption experiment.

    verified intermediate checkpoint -> restore weights, AdamW, RNG
                                      -> train to final step
                                      -> recovered.pt
                                      -> exact comparison with uninterrupted.pt

Importantly, we also hash-check the original uninterrupted checkpoint
before treating it as a comparison reference. A mismatching reference
is a hard error.

Run the audit in lesson 257 first, then:
    python examples/154_replay_verified_restart.py

**Experiment:** Damage a COPY of the newest intermediate stage.
The replay should fall back to an earlier verified one.

**Checkpoint:** Why must the learning-rate schedule still target the
original total step count after choosing an earlier restart point?
