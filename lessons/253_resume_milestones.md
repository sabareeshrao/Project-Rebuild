# Lesson 253: several interruptions, one training trajectory

The previous resume experiment stopped after two steps and continued
to step four. This lesson generalizes the schedule to any ordered
sequence of stopping points.

For total_steps=6 and stops=(2, 4), the segments are:

    steps 1-2 -> save -> steps 3-4 -> save -> steps 5-6

The points refer to completed optimizer steps, not token positions.
A repeated, decreasing or out-of-range stopping point is rejected.

Run: python examples/146_resume_milestones.py

Experiment: Plan an 8-step run interrupted after steps 1, 3 and 7.
Which segment contains the most optimizer updates?

Checkpoint: Why must the final target step be strictly larger than
every intermediate stopping point?
