"""
How long each junction leaves a movement unserved, measured from the traces.

WHY THIS EXISTS
---------------
The single-intersection study's Addendum E found that its unconstrained
policies starved a movement for 245-340 s against a 200 s bound, and that
nothing in the headline numbers hinted at it. The reason is structural: a
starved approach carries few enough vehicles that its misery is diluted by the
ones being served well, so mean waiting time cannot show it. The evidence has
to be the signal's own behaviour.

This module reads the `signal_<junction>` columns the corridor's episode runner
records and reports, per controller and per junction:

    longest_green_s        the longest single continuous green
    longest_unserved_s     the longest a phase went without being shown green
    phases_seen            how many distinct green phases ever appeared

`phases_seen` is the blunt one. A junction that only ever shows three of its
four phases has starved the fourth completely, and that is visible at a glance.

WHAT THIS IS NOT
----------------
It is a measurement of these runs, not a guarantee about all of them. The
guarantee would come from the `+safe` wrapper, which is available and tested
but was not swept here. So this answers "did it happen on the hours we
measured", which is a weaker and more honest claim than "it cannot happen".
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corridor import paths  # noqa: E402
from corridor.evaluate_corridor import CONTROLLER_ORDER, LABELS  # noqa: E402
from eval.simulation import DELTA_TIME  # noqa: E402


def signal_columns(header: list) -> list:
    return [c for c in header if c.startswith("signal_")]


def analyse_trace(path: Path) -> dict:
    """Per-junction green/unserved extremes from one trace file."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}

    out = {}
    for column in signal_columns(list(rows[0])):
        junction = column[len("signal_"):]
        seen, last_green = set(), None
        green_run, longest_green = 0, 0
        since_served = {}
        longest_unserved = 0

        for row in rows:
            state = row[column]
            if not state:
                continue
            is_green = "y" not in state and ("G" in state or "g" in state)
            if is_green:
                seen.add(state)
                # Every phase seen so far that is NOT the one showing now has
                # been waiting another decision interval.
                for phase in seen:
                    if phase == state:
                        since_served[phase] = 0
                    else:
                        since_served[phase] = since_served.get(phase, 0) + DELTA_TIME
                        longest_unserved = max(longest_unserved,
                                               since_served[phase])
                green_run = green_run + DELTA_TIME if state == last_green else DELTA_TIME
                longest_green = max(longest_green, green_run)
                last_green = state
            else:
                # Amber: no phase has right of way, so it neither extends the
                # current green nor counts as service for anyone.
                green_run, last_green = 0, None
                for phase in seen:
                    since_served[phase] = since_served.get(phase, 0) + DELTA_TIME
                    longest_unserved = max(longest_unserved, since_served[phase])

        out[junction] = {
            "phases_seen": len(seen),
            "longest_green_s": longest_green,
            "longest_unserved_s": longest_unserved,
        }
    return out


def summarise(n_junctions: int = 3, scenario: str = "calibrated") -> list:
    """One row per controller: the worst junction on each measure."""
    rows = []
    for controller in CONTROLLER_ORDER:
        per_junction = analyse_trace(
            paths.trace_file(controller, scenario, n_junctions))
        if not per_junction:
            continue
        rows.append({
            "controller": controller,
            "label": LABELS.get(controller, controller),
            "junctions": len(per_junction),
            "min_phases_seen": min(v["phases_seen"] for v in per_junction.values()),
            "longest_green_s": max(v["longest_green_s"] for v in per_junction.values()),
            "longest_unserved_s": max(v["longest_unserved_s"]
                                      for v in per_junction.values()),
        })
    return rows


def table(n_junctions: int = 3, scenario: str = "calibrated") -> str:
    """Markdown for CORRIDOR_REPORT.md's phase-service section."""
    rows = summarise(n_junctions, scenario)
    if not rows:
        return None
    from eval.simulation import SafetyConstrainedController as S

    out = ["| Controller | Green phases ever shown | Longest green (s) | "
           "Longest a phase went unserved (s) |", "|---|---|---|---|"]
    for row in rows:
        out.append("| %s | %d of 4 | %d | %d |"
                   % (row["label"], row["min_phases_seen"],
                      row["longest_green_s"], row["longest_unserved_s"]))
    out.append("")
    out.append("Measured on `%s`, worst junction of %d, one evaluation hour "
               "each. For reference the single-intersection study's bounds are "
               "a %d s maximum green and a %d s starvation bound."
               % (scenario, rows[0]["junctions"], S.MAX_GREEN, S.MAX_STARVATION))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure phase service from the corridor traces.")
    parser.add_argument("--junctions", type=int, default=3)
    parser.add_argument("--scenario", default="calibrated")
    args = parser.parse_args()

    text = table(args.junctions, args.scenario)
    if text is None:
        print("no traces found - run the evaluation first")
        return
    print(text)


if __name__ == "__main__":
    main()
