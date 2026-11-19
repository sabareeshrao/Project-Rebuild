# Lesson 294 — Every numerical comparison has a tolerance

A JSON report saying `"passed"` is not enough to answer *how close*
the compared outputs were. Examine the **actual numerical errors**.

Our new inspector checks saved causal, batch, training/inference and
attention differences against explicit absolute limits. Finite
differences have a separate, looser limit because they approximate
a mathematical derivative using subtraction.

```text
saved eight-check report
     -> validated measurements
     -> compare error to a documented budget
     -> report the observed errors and limits
```

Run the release review (lesson 292), then
`python examples/187_conformance_budgets.py`.

Try lowering `max_logit_error` to a smaller value. Does the test
begin failing due to floating-point roundoff, and is that a model bug
or a tolerance-policy decision?

This lesson does not change model weights or claim GPT-2 quality.
