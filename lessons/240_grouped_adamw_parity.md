# Lesson 240: compare the original optimizer, not two cloned optimizers

The earlier training-parity lesson used identical plain AdamW
instances. Here, **our configure_adamw** is compared with the
reference's actual `configure_optimizers` implementation.

```text
matching weights -> matching gradients
     |                    |
local grouped AdamW   upstream grouped AdamW
     |                    |
     +-- verify decay membership --+
     |                              |
     +-- step and compare weights --+
```

We check the tensor names and their assigned weight decay before
updating any parameters. We then compare every model tensor after
one step. The experiment changes in-memory model objects only.

Run after producing `out/capstone/model.pt`:
```bash
python examples/133_grouped_adamw_parity.py \
  --upstream-model /trusted/nanoGPT/model.py
```

The reference file executes Python code, so review it before use.
The local tests use a test double and do not prove original parity.

**Checkpoint:** Why is matching the optimizer's *group membership*
different from using the same learning rate and AdamW class?
