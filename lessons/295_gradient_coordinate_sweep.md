# Lesson 295 — One correct gradient doesn't validate all of backprop

A finite-difference check at one token-embedding weight gives useful
evidence, but different parameter groups can have distinct bugs.

This experiment checks **three scalar derivatives** at two important
sites: the token lookup and positional embedding tables.

```text
loss -> d/d(token embedding [0,0])
     -> d/d(token embedding [1,0])
     -> d/d(position embedding [0,0])
each -> autograd vs central finite difference
```

All calculations are on a float64 CPU model, with dropout disabled.
The checker restores weights and modes between observations.

Run `python examples/188_multi_gradient_check.py`.

**Exercise:** Deliberately perturb only the positional embedding's
gradient in a test hook. Why might the original token-only numerical
check still pass while this broader sweep fails?
