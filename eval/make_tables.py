"""
Render the evaluation results as Markdown tables for REPORT.md.

Run (after eval/evaluate.py):
    python eval/make_tables.py              # print to stdout
    python eval/make_tables.py --write      # also save results/logs/results_tables.md

WHY GENERATE THE TABLES INSTEAD OF TYPING THEM
----------------------------------------------
Every number in the report comes out of evaluation_summary.csv through this
script. Copying figures by hand into a write-up is where reports quietly go
wrong - a digit gets transposed, or a table is updated after a re-run but a
sentence three paragraphs down is not. Generating them means the report cannot
disagree with the data it is describing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOGS_DIR = ROOT / "results" / "logs"

ORDER = ["fixed-time", "fixed-time-tuned", "greedy-queue", "dqn", "ppo"]
LABELS = {
    "fixed-time": "Fixed-time (untuned)",
    "fixed-time-tuned": "Fixed-time (Webster-tuned)",
    "greedy-queue": "Greedy queue heuristic",
    "dqn": "**DQN (learned)**",
    "ppo": "PPO (learned)",
}
MAIN = ["balanced", "ns_heavy", "peak"]
ROBUST = ["ew_heavy", "surge", "light"]


def read_summary() -> list:
    path = LOGS_DIR / "evaluation_summary.csv"
    if not path.exists():
        raise SystemExit("No evaluation_summary.csv - run:  python eval/evaluate.py")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def value(summary: list, controller: str, scenario: str, field: str):
    for row in summary:
        if row["controller"] == controller and row["scenario"] == scenario:
            try:
                return float(row[field])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def controllers_in(summary: list) -> list:
    found = {r["controller"] for r in summary}
    return [c for c in ORDER if c in found]


def scenario_table(summary: list, scenario: str) -> str:
    """One table: every controller's metrics on a single demand scenario."""
    lines = [
        "| Controller | Mean wait (s) | Mean queue (veh) | Time lost (s) | "
        "Throughput (veh/h) | Wait vs fixed-time |",
        "|---|---|---|---|---|---|",
    ]
    base = value(summary, "fixed-time", scenario, "mean_waiting_time_s_mean")
    for controller in controllers_in(summary):
        wait = value(summary, controller, scenario, "mean_waiting_time_s_mean")
        if wait is None:
            continue
        wait_sd = value(summary, controller, scenario, "mean_waiting_time_s_std")
        queue = value(summary, controller, scenario, "mean_queue_length_veh_mean")
        queue_sd = value(summary, controller, scenario, "mean_queue_length_veh_std")
        loss = value(summary, controller, scenario, "mean_time_loss_s_mean")
        through = value(summary, controller, scenario, "throughput_veh_mean")

        if controller == "fixed-time" or not base:
            delta = "&mdash;"
        else:
            pct = 100.0 * (wait - base) / base
            delta = "**%+.1f%%**" % pct if controller == "dqn" else "%+.1f%%" % pct

        lines.append("| %s | %.1f ± %.1f | %.1f ± %.1f | %.1f | %.0f | %s |"
                     % (LABELS[controller], wait, wait_sd, queue, queue_sd,
                        loss, through, delta))
    return "\n".join(lines)


def headline_table(summary: list, scenarios: list) -> str:
    """Waiting time for every controller across several scenarios, side by side."""
    header = "| Controller | " + " | ".join(scenarios) + " |"
    divider = "|---" * (len(scenarios) + 1) + "|"
    lines = [header, divider]
    for controller in controllers_in(summary):
        cells = []
        for scenario in scenarios:
            wait = value(summary, controller, scenario, "mean_waiting_time_s_mean")
            cells.append("&mdash;" if wait is None else "%.1f s" % wait)
        lines.append("| %s | %s |" % (LABELS[controller], " | ".join(cells)))
    return "\n".join(lines)


def improvement_table(summary: list, scenarios: list) -> str:
    """Reduction in waiting time versus the untuned fixed-time plan."""
    header = "| Controller | " + " | ".join(scenarios) + " |"
    divider = "|---" * (len(scenarios) + 1) + "|"
    lines = [header, divider]
    for controller in controllers_in(summary):
        if controller == "fixed-time":
            continue
        cells = []
        for scenario in scenarios:
            base = value(summary, "fixed-time", scenario, "mean_waiting_time_s_mean")
            wait = value(summary, controller, scenario, "mean_waiting_time_s_mean")
            if base and wait is not None:
                cells.append("%+.1f%%" % (100.0 * (wait - base) / base))
            else:
                cells.append("&mdash;")
        lines.append("| %s | %s |" % (LABELS[controller], " | ".join(cells)))
    return "\n".join(lines)


def build(summary: list) -> str:
    present_main = [s for s in MAIN if any(r["scenario"] == s for r in summary)]
    present_robust = [s for s in ROBUST if any(r["scenario"] == s for r in summary)]

    out = ["## Generated results tables", ""]
    out += ["### Mean waiting time per vehicle, by scenario", "",
            headline_table(summary, present_main + present_robust), ""]
    out += ["### Change in waiting time vs the untuned fixed-time baseline "
            "(negative is better)", "",
            improvement_table(summary, present_main + present_robust), ""]
    for scenario in present_main + present_robust:
        out += ["### Scenario: `%s`" % scenario, "",
                scenario_table(summary, scenario), ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--write", action="store_true",
                        help="also write results/logs/results_tables.md")
    args = parser.parse_args()

    text = build(read_summary())
    print(text)
    if args.write:
        path = LOGS_DIR / "results_tables.md"
        path.write_text(text, encoding="utf-8")
        print("\nWrote " + str(path))


if __name__ == "__main__":
    main()
