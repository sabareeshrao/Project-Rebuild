# Lesson 258: a successful audit must have a coherent structure

A JSON document can claim `"status": "passed"` without proving
anything. Before relying on that claim, check its format version,
step schedule, stage count and SHA-256 syntax.

The stages must match the declared plan. A six-step run interrupted
at steps 2 and 4 must contain stage records for 2, 4 and 6.

    JSON audit -> validate schema -> validate stage order
                                   -> validate digest syntax

Run `python examples/150_multistage_resume_audit.py` first, then
`python examples/151_read_resume_manifest.py`.

**Experiment:** Change a stage's step from 4 to 3 in a COPY of
the audit. Why should the schema validator reject it?

**Important:** An internally consistent JSON manifest is not
cryptographic proof that its files are unchanged. The next lesson
will hash those actual files, too.
