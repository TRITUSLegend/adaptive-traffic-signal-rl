"""
Where the evaluation writes its files, and what it calls them.

WHY THIS MODULE EXISTS
----------------------
The per-episode traces are written by `eval/evaluate.py` and read back by two
different scripts - `eval/make_plots.py` for the queue-over-time figure and
`eval/followups.py` for the safety bounds in addendum E. Each of them used to
build the filename itself, and they did not agree: the writer turned
`greedy-queue+safe` into `trace_greedy_queue+safe_<scenario>.csv` while the
reader looked for `trace_greedy_queue_safe_<scenario>.csv`.

Nothing raised. `trace_bounds` returns None for a file it cannot find and the
addendum simply omitted the row, so the safety table would have been published
with the wrapped controllers silently missing - the exact rows the addendum
exists to show. One function, imported by everyone, is the fix.

Deliberately free of heavy imports: `make_plots.py` must not have to pull in
SUMO to name a file.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "results" / "logs"
PLOTS_DIR = ROOT / "results" / "plots"


def trace_filename(controller: str, scenario: str) -> str:
    """
    The trace file for one (controller, scenario) pair.

    Hyphens become underscores so the names read as identifiers; the `+` of a
    `+safe` suffix is kept, because it is what distinguishes the constrained
    run from the unconstrained one and flattening it to `_` would make
    `dqn+safe` indistinguishable from a model tagged `safe`.
    """
    return "trace_%s_%s.csv" % (controller.replace("-", "_"), scenario)


def trace_path(controller: str, scenario: str) -> Path:
    """Absolute path of the trace file for one (controller, scenario) pair."""
    return LOGS_DIR / trace_filename(controller, scenario)
