"""
Multi-intersection corridor extension.

Everything in this package is NEW work added for the corridor study. It does
not modify the single-intersection project: it may *import* from `sumo_setup`
and `eval.simulation` (read-only reuse of the Controller interface and the
metric helpers), but it never writes to `networks/`, `agent/models/`,
`results/logs/` or `REPORT.md`.

`tests/test_corridor_isolation.py` enforces that by hashing the
single-intersection source files, so an accidental edit fails the suite rather
than being noticed after it has changed a published number.
"""
