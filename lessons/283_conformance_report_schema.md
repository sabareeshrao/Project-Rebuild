# Lesson 283 — Treat a conformance report as a claim

Any JSON file can contain `"status": "passed"`. That does not prove
the model was correct. We first verify the *structure* of the saved
evidence: eight expected checks, finite nonnegative errors, genuine
boolean results, source digests, model dimensions and consistent seeds.

```text
conformance.json -> parse -> eight named checks?
                           -> valid measured errors?
                           -> consistent vocab and seed?
                           -> SHA-256 digests present?
```

Run the extended audit from lesson 282, then
`python examples/176_read_conformance.py`.

**Exercise:** Change one check's `repeatable` field to `false` in
a copy of the JSON. Why must the validator reject the report?

Schema validation is not an authenticity guarantee: the next lesson
will compare the report's digests with the actual saved files.
