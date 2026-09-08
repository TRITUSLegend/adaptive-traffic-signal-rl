"""
Render the evaluation results as Markdown tables for REPORT.md.

Run (after eval/evaluate.py):
    python eval/make_tables.py              # print to stdout
    python eval/make_tables.py --write      # save the tables AND update REPORT.md

WHY GENERATE THE TABLES INSTEAD OF TYPING THEM
----------------------------------------------
Every number in the report comes out of evaluation_summary.csv through this
script. Copying figures by hand into a write-up is where reports quietly go
wrong - a digit gets transposed, or a table is updated after a re-run but a
sentence three paragraphs down is not. Generating them means the report cannot
disagree with the data it is describing.

HOW THE REPORT IS KEPT IN SYNC
------------------------------
`--write` does two things:

  1. writes results/logs/results_tables.md - the full per-scenario appendix;
  2. splices the four tables REPORT.md displays into REPORT.md itself, between
     paired `<!-- BEGIN GENERATED: name -->` / `<!-- END GENERATED: name -->`
     markers.

Point 2 is what makes the claim in REPORT.md section 8 true. Before it existed
the script only wrote the appendix, and the report's own tables were
hand-transcribed - which happened to be accurate, but was exactly the
unguarded step the appendix was supposed to protect against. If a marker is
missing or unpaired the splice fails loudly rather than writing a half-updated
report.
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

# Column headings REPORT.md uses for the two headline tables. They carry the
# demand context the report's narrative refers to, so they live with the
# generator rather than being retyped into the document.
SCENARIO_HEADINGS = {
    "balanced": "`balanced`",
    "ns_heavy": "`ns_heavy` *(trained)*",
    "peak": "`peak`",
    "ew_heavy": "`ew_heavy` (mirrored)",
    "surge": "`surge` (+30%)",
    "light": "`light` (−40%)",
}

REPORT_PATH = ROOT / "REPORT.md"


def read_summary() -> list:
    path = LOGS_DIR / "evaluation_summary.csv"
    if not path.exists():
        raise SystemExit("No evaluation_summary.csv - run:  python eval/evaluate.py")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_raw() -> list:
    """
    Per-episode rows. Needed for the insertion-backlog table, whose figure is a
    mean across seeds and so cannot be read off the summary (which keeps only
    the worst case).
    """
    path = LOGS_DIR / "evaluation_raw.csv"
    if not path.exists():
        raise SystemExit("No evaluation_raw.csv - run:  python eval/evaluate.py")
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


# ---------------------------------------------------------------------------
# The four tables REPORT.md itself displays
# ---------------------------------------------------------------------------
def best_controller(summary: list, scenario: str) -> str:
    """Controller with the lowest mean waiting time on one scenario."""
    ranked = []
    for controller in controllers_in(summary):
        wait = value(summary, controller, scenario, "mean_waiting_time_s_mean")
        if wait is not None:
            ranked.append((wait, controller))
    return min(ranked)[1] if ranked else ""


def report_wait_table(summary: list, scenarios: list) -> str:
    """
    Mean wait ± one standard deviation, best controller per column in bold.

    This is the format sections 4.2 and 4.4 of the report use: the spread
    matters as much as the mean there, because several of the gaps being
    discussed are only a second or two wide.
    """
    headings = [SCENARIO_HEADINGS.get(s, "`%s`" % s) for s in scenarios]
    lines = ["| Controller | " + " | ".join(headings) + " |",
             "|---" * (len(scenarios) + 1) + "|"]
    best = {s: best_controller(summary, s) for s in scenarios}

    for controller in controllers_in(summary):
        cells = []
        for scenario in scenarios:
            wait = value(summary, controller, scenario, "mean_waiting_time_s_mean")
            sd = value(summary, controller, scenario, "mean_waiting_time_s_std")
            if wait is None:
                cells.append("&mdash;")
                continue
            cell = "%.1f ± %.1f" % (wait, sd or 0.0)
            cells.append("**%s**" % cell if best[scenario] == controller else cell)
        lines.append("| %s | %s |" % (LABELS[controller], " | ".join(cells)))
    return "\n".join(lines)


def heuristic_table(summary: list, scenarios: list) -> str:
    """
    The learned agent against the greedy heuristic, scenario by scenario.

    The percentage is always expressed relative to the WORSE of the two, so
    "25% better" means the winner's wait is 25% below the loser's. Stating the
    direction matters: the same pair of numbers gives a different figure if the
    denominator is the winner instead, and mixing the two is how a comparison
    table quietly overstates a result.
    """
    lines = ["| Scenario | Greedy | DQN | Winner |", "|---|---|---|---|"]
    for scenario in scenarios:
        greedy = value(summary, "greedy-queue", scenario, "mean_waiting_time_s_mean")
        dqn = value(summary, "dqn", scenario, "mean_waiting_time_s_mean")
        if greedy is None or dqn is None:
            continue
        winner = "DQN" if dqn < greedy else "Greedy"
        worse, better = max(dqn, greedy), min(dqn, greedy)
        pct = 100.0 * (worse - better) / worse
        verdict = "%s (%.0f%% better)" % (winner, pct)
        # Bold the cases the report's argument turns on: where the learned
        # policy is not merely ahead but decisively so.
        if winner == "DQN" and pct >= 20:
            verdict = "**%s**" % verdict
        lines.append("| `%s` | %.1f | %.1f | %s |" % (scenario, greedy, dqn, verdict))
    return "\n".join(lines)


def insertion_backlog_table(raw: list) -> str:
    """
    Vehicles still queued for insertion when the hour ended, averaged over seeds.

    This is the honesty check. A vehicle SUMO never managed to place on the
    network accrues no waiting time, so it silently improves the average of
    whichever controller was too congested to admit it.

    Note what the number is: a snapshot of the insertion backlog at t=3600, not
    a count accumulated over the hour. The two nearly coincide here because
    demand stops when the episode does, but the snapshot is what is measured
    and so is what the column says.

    Restricted to the five controllers the report compares, like every other
    generated table. `evaluation_raw.csv` also carries the addendum variants -
    extra training seeds, the mixed-demand agent, the `+safe` wrappers - and
    they belong in their own addendum, not in a section 5 table that a reader
    will match against the five rows above it. It is also what stops an
    unrecognised controller from reaching the `LABELS` lookup below.
    """
    grouped = {}
    for row in raw:
        if row["controller"] not in LABELS:
            continue
        key = (row["scenario"], row["controller"])
        grouped.setdefault(key, []).append(float(row["vehicles_not_inserted"]))

    rows = []
    clean = []
    for (scenario, controller), values in grouped.items():
        mean = sum(values) / len(values)
        if any(v < 0 for v in values):
            # -1 is the sentinel run_episode stores when the backlog could not
            # be read at all; a mean over it would be meaningless.
            rows.append((float("inf"), scenario, controller, "not measured"))
        elif mean >= 0.5:
            rows.append((mean, scenario, controller, "%.0f" % mean))
        else:
            clean.append(controller)

    rows.sort(key=lambda r: -r[0])
    lines = ["| Scenario | Controller | Still awaiting insertion at the "
             "hour's end (mean) |", "|---|---|---|"]
    for mean, scenario, controller, text in rows:
        label = LABELS[controller].replace("**", "")
        # The two that matter for the argument are an order of magnitude larger
        # than the rest; bold them so the table reads at a glance.
        cell = "**%s**" % text if mean >= 50 else text
        lines.append("| `%s` | %s | %s |" % (scenario, label, cell))
    if clean:
        names = [LABELS[c].replace("**", "") for c in ORDER if c in set(clean)]
        lines.append("| all others | %s | 0 |" % ", ".join(names))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Splicing the tables into REPORT.md
# ---------------------------------------------------------------------------
def splice(text: str, name: str, block: str) -> str:
    """
    Replace one marked region of REPORT.md with freshly generated content.

    Fails loudly on a missing or unpaired marker. A silent skip here would be
    the worst outcome: the report would keep its old numbers while the script
    reported success, which is the exact failure this whole mechanism exists
    to prevent.
    """
    begin = "<!-- BEGIN GENERATED: %s -->" % name
    end = "<!-- END GENERATED: %s -->" % name
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(
            "REPORT.md: expected exactly one %s ... %s marker pair.\n"
            "Found %d begin and %d end markers."
            % (begin, end, text.count(begin), text.count(end)))
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return head + begin + "\n" + block + "\n" + end + tail


def update_report(summary: list, raw: list) -> list:
    """Rewrite REPORT.md's generated regions. Returns the region names written."""
    if not REPORT_PATH.exists():
        raise SystemExit("REPORT.md not found next to this repository's root")

    present_main = [s for s in MAIN if any(r["scenario"] == s for r in summary)]
    present_robust = [s for s in ROBUST if any(r["scenario"] == s for r in summary)]

    regions = [
        ("main-wait", report_wait_table(summary, present_main)),
        ("robustness-wait", report_wait_table(summary, present_robust)),
        ("heuristic-vs-dqn",
         heuristic_table(summary, present_main + present_robust)),
        ("insertion-backlog", insertion_backlog_table(raw)),
    ]

    text = REPORT_PATH.read_text(encoding="utf-8")
    for name, block in regions:
        text = splice(text, name, block)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return [name for name, _ in regions]


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
                        help="write results/logs/results_tables.md and splice "
                             "the report's own tables into REPORT.md")
    args = parser.parse_args()

    summary = read_summary()
    text = build(summary)
    print(text)
    if args.write:
        path = LOGS_DIR / "results_tables.md"
        path.write_text(text, encoding="utf-8")
        print("\nWrote " + str(path))
        written = update_report(summary, read_raw())
        print("Updated %s (%s)" % (REPORT_PATH.name, ", ".join(written)))


if __name__ == "__main__":
    main()
