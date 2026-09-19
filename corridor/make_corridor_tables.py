"""
Generate CORRIDOR_REPORT.md's tables from the evaluation CSVs.

Run:
    python corridor/make_corridor_tables.py            # print them
    python corridor/make_corridor_tables.py --write    # splice into the report

WHY THE TABLES ARE GENERATED
----------------------------
Defect D4 in the single-intersection study: the report claimed its numbers came
from the CSVs, and they did - the first time they were typed. Nothing kept them
there. Generated regions plus `tests/test_corridor_report_matches_data.py`
close that loop, so a number in the report cannot drift from the data it
describes without failing the suite.

Each region is delimited by

    <!-- BEGIN GENERATED: name -->  ...  <!-- END GENERATED: name -->

and only the text between the markers is replaced, so the surrounding narrative
is never touched by this script.

THROUGHPUT TRAVELS WITH WAITING TIME
------------------------------------
Every table that reports delay reports throughput in the same row. A controller
can make waiting time look excellent by refusing to admit vehicles, and on a
corridor it can also do so by starving the link that feeds its neighbour. The
project's rule is that the two numbers are never separated, so there is no
table here from which throughput has been dropped for width.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corridor import paths  # noqa: E402
from corridor.evaluate_corridor import CONTROLLER_ORDER, LABELS  # noqa: E402
from corridor.perturbations import MODES  # noqa: E402
from corridor.robustness import FAILURE_RATES  # noqa: E402

REPORT_PATH = paths.ROOT / "CORRIDOR_REPORT.md"
SCENARIO_ORDER = ["calibrated", "balanced", "peak", "reverse"]
BASELINE = "fixed-time"


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def read(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cell(rows: list, controller: str, scenario: str, metric: str,
         network: str = None):
    """Mean and sample sd of one metric over the seeds of one cell."""
    values = [float(r[metric]) for r in rows
              if r["controller"] == controller and r["scenario"] == scenario
              and (network is None or r.get("network") == network)]
    if not values:
        return None, None
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), sd


def perturbed_cell(rows: list, controller: str, rate: float, mode: str,
                   metric: str):
    """
    One cell of the failure sweep.

    The clean control (rate 0) is stored once with mode 'none' and belongs to
    every mode's column, so it is matched on rate alone.
    """
    values = []
    for row in rows:
        if row["controller"] != controller:
            continue
        if abs(float(row["failure_rate"]) - rate) > 1e-9:
            continue
        if rate > 0 and row["failure_mode"] != mode:
            continue
        values.append(float(row[metric]))
    if not values:
        return None
    return statistics.mean(values)


def fmt(value, places: int = 1) -> str:
    return "-" if value is None else ("%.*f" % (places, value))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def headline_table(rows: list) -> str:
    """Waiting time and throughput per controller per scenario."""
    if not rows:
        return None
    out = ["| Controller | Scenario | Mean wait (s) | Queue (veh) | Throughput (veh/h) | Not inserted |",
           "|---|---|---|---|---|---|"]
    for scenario in SCENARIO_ORDER:
        for controller in CONTROLLER_ORDER:
            wait, wait_sd = cell(rows, controller, scenario, "mean_waiting_time_s")
            if wait is None:
                continue
            queue, _ = cell(rows, controller, scenario, "mean_queue_length_veh")
            through, _ = cell(rows, controller, scenario, "throughput_veh")
            backlog, _ = cell(rows, controller, scenario, "vehicles_not_inserted")
            name = LABELS.get(controller, controller)
            if controller != BASELINE:
                name = "  " + name
            out.append("| %s | `%s` | %s ± %s | %s | %s | %s |"
                       % (name, scenario, fmt(wait), fmt(wait_sd),
                          fmt(queue), fmt(through, 0), fmt(backlog, 0)))
    return "\n".join(out)


def reduction_table(rows: list) -> str:
    """Delay reduction against fixed-time - the brief's headline metric."""
    if not rows:
        return None
    out = ["| Controller | " + " | ".join("`%s`" % s for s in SCENARIO_ORDER)
           + " | Mean |", "|---|" + "---|" * (len(SCENARIO_ORDER) + 1)]
    for controller in CONTROLLER_ORDER:
        if controller == BASELINE:
            continue
        cells, reductions = [], []
        for scenario in SCENARIO_ORDER:
            base, _ = cell(rows, BASELINE, scenario, "mean_waiting_time_s")
            value, _ = cell(rows, controller, scenario, "mean_waiting_time_s")
            if base is None or value is None or base == 0:
                cells.append("-")
                continue
            change = 100.0 * (value - base) / base
            reductions.append(change)
            cells.append("%+.1f%%" % change)
        mean = ("%+.1f%%" % statistics.mean(reductions)) if reductions else "-"
        out.append("| %s | %s | **%s** |"
                   % (LABELS.get(controller, controller), " | ".join(cells), mean))
    return "\n".join(out)


def throughput_table(rows: list) -> str:
    """
    Throughput on its own, so the delay reductions above can be read against it.

    A reduction bought by admitting fewer vehicles is not a reduction, and this
    is the table that makes that visible rather than asserted.
    """
    if not rows:
        return None
    out = ["| Controller | " + " | ".join("`%s`" % s for s in SCENARIO_ORDER)
           + " |", "|---|" + "---|" * len(SCENARIO_ORDER)]
    for controller in CONTROLLER_ORDER:
        cells = []
        for scenario in SCENARIO_ORDER:
            through, _ = cell(rows, controller, scenario, "throughput_veh")
            cells.append(fmt(through, 0))
        out.append("| %s | %s |" % (LABELS.get(controller, controller),
                                    " | ".join(cells)))
    return "\n".join(out)


def convergence_table() -> str:
    """Where each learned controller started and finished in training."""
    runs = [("idqn", "Independent DQN"), ("ippo", "Independent PPO"),
            ("ma2c", "MA2C")]
    out = ["| Run | Episodes | Mean queue, first 10 (veh) | Mean queue, last 10 (veh) | Change |",
           "|---|---|---|---|---|"]
    any_row = False
    for name, label in runs:
        curve = read(paths.training_curve(name))
        if not curve:
            continue
        any_row = True
        queues = [float(r["mean_queue_veh"]) for r in curve]
        first = statistics.mean(queues[:10])
        last = statistics.mean(queues[-10:])
        out.append("| %s | %d | %.1f | %.1f | %+.0f%% |"
                   % (label, len(curve), first, last,
                      100.0 * (last - first) / first if first else 0.0))
    return "\n".join(out) if any_row else None


def detector_table(rows: list) -> str:
    """Waiting time at each detector-failure rate, per mode."""
    if not rows:
        return None
    blocks = []
    for mode in MODES:
        lines = ["**Failure mode: %s**" % ("stuck-at-zero" if mode == "zero"
                                           else "freeze (last value held)"),
                 "",
                 "| Controller | " + " | ".join(
                     "%.0f%% dead" % (100 * r) for r in FAILURE_RATES)
                 + " | Change at %.0f%% |" % (100 * max(FAILURE_RATES)),
                 "|---|" + "---|" * (len(FAILURE_RATES) + 1)]
        for controller in CONTROLLER_ORDER:
            # `heaviest` is the value at the LARGEST failure rate, which is the
            # quantity the column reports. It is deliberately not the maximum
            # over rates: "how bad does it get when half the detectors die" is
            # a different and more answerable question than "what is the worst
            # cell anywhere in this row", and conflating them would let a noisy
            # middle rate set the headline.
            cells, clean, heaviest = [], None, None
            for rate in FAILURE_RATES:
                value = perturbed_cell(rows, controller, rate, mode,
                                       "mean_waiting_time_s")
                cells.append(fmt(value))
                if rate == 0.0:
                    clean = value
                if rate == max(FAILURE_RATES):
                    heaviest = value
            if clean and heaviest:
                change = "%+.0f%%" % (100.0 * (heaviest - clean) / clean)
            else:
                change = "-"
            lines.append("| %s | %s | **%s** |"
                         % (LABELS.get(controller, controller),
                            " | ".join(cells), change))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def margin_table(rows: list) -> str:
    """
    The brief's actual question: does the adaptive advantage survive the fault?

    Reports each controller's waiting time MINUS fixed-time's at the same
    failure rate. Negative means still ahead of the clock. Fixed-time reads no
    detectors, so its own row is flat by construction and is the reference.
    """
    if not rows:
        return None
    out = ["| Controller | " + " | ".join("%.0f%% dead" % (100 * r)
                                          for r in FAILURE_RATES) + " |",
           "|---|" + "---|" * len(FAILURE_RATES)]
    for controller in CONTROLLER_ORDER:
        if controller == BASELINE:
            continue
        cells = []
        for rate in FAILURE_RATES:
            mode = "zero"
            base = perturbed_cell(rows, BASELINE, rate, mode,
                                  "mean_waiting_time_s")
            value = perturbed_cell(rows, controller, rate, mode,
                                   "mean_waiting_time_s")
            if base is None or value is None:
                cells.append("-")
            else:
                cells.append("%+.1f s" % (value - base))
        out.append("| %s | %s |" % (LABELS.get(controller, controller),
                                    " | ".join(cells)))
    return "\n".join(out)


def transfer_table(rows: list) -> str:
    """Zero-shot performance on networks never trained on."""
    if not rows:
        return None
    networks = []
    for row in rows:
        if row["network"] not in networks:
            networks.append(row["network"])

    out = ["| Controller | " + " | ".join("`%s`" % n for n in networks) + " |",
           "|---|" + "---|" * len(networks)]
    for controller in CONTROLLER_ORDER:
        cells = []
        for network in networks:
            values = [float(r["mean_waiting_time_s"]) for r in rows
                      if r["controller"] == controller
                      and r["network"] == network]
            through = [float(r["throughput_veh"]) for r in rows
                       if r["controller"] == controller
                       and r["network"] == network]
            if not values:
                cells.append("-")
            else:
                cells.append("%.1f s / %.0f veh"
                             % (statistics.mean(values), statistics.mean(through)))
        out.append("| %s | %s |" % (LABELS.get(controller, controller),
                                    " | ".join(cells)))
    return "\n".join(out)



def ma2c_protocol_table(rows: list) -> str:
    """
    The same trained MA2C policy, read two ways.

    Deterministic is the protocol every other controller is held to. Sampled is
    what the agent did while training. They differ enormously here, and the
    reason is a property of the policy rather than of the algorithm - see the
    phase-service table below it.
    """
    if not rows:
        return None
    pairs = [("ma2c", "Deterministic (argmax) - the protocol used for every "
                      "other controller"),
             ("ma2c-sampled", "Sampled from the policy - what it did while "
                              "training")]
    if not any(r["controller"] == "ma2c-sampled" for r in rows):
        return None

    out = ["| Action rule | " + " | ".join("`%s`" % s for s in SCENARIO_ORDER)
           + " | Throughput |", "|---|" + "---|" * (len(SCENARIO_ORDER) + 1)]
    for controller, label in pairs:
        cells, through = [], []
        for scenario in SCENARIO_ORDER:
            wait, _ = cell(rows, controller, scenario, "mean_waiting_time_s")
            thr, _ = cell(rows, controller, scenario, "throughput_veh")
            cells.append(fmt(wait))
            if thr is not None:
                through.append(thr)
        mean_thr = statistics.mean(through) if through else None
        out.append("| %s | %s | %s |" % (label, " | ".join(cells),
                                         fmt(mean_thr, 0)))
    # Fixed-time for reference, since the question is whether either beats it.
    cells = [fmt(cell(rows, BASELINE, s, "mean_waiting_time_s")[0])
             for s in SCENARIO_ORDER]
    thr = [cell(rows, BASELINE, s, "throughput_veh")[0] for s in SCENARIO_ORDER]
    out.append("| *Fixed-time, for reference* | %s | %s |"
               % (" | ".join(cells), fmt(statistics.mean(thr), 0)))
    return "\n".join(out)


def phase_service_table(rows: list) -> str:
    """What each controller's signals actually did, read from the traces."""
    from corridor.phase_service import table as service_table
    return service_table()


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------
BUILDERS = [
    ("headline", lambda d: headline_table(d["eval"])),
    ("reduction", lambda d: reduction_table(d["eval"])),
    ("throughput", lambda d: throughput_table(d["eval"])),
    ("convergence", lambda d: convergence_table()),
    ("detector", lambda d: detector_table(d["robust"])),
    ("margin", lambda d: margin_table(d["robust"])),
    ("transfer", lambda d: transfer_table(d["transfer"])),
    ("ma2c-protocol", lambda d: ma2c_protocol_table(d["eval"])),
    ("phase-service", lambda d: phase_service_table(d["eval"])),
]


def splice(text: str, name: str, body: str) -> str:
    begin = "<!-- BEGIN GENERATED: %s -->" % name
    end = "<!-- END GENERATED: %s -->" % name
    if begin not in text or end not in text:
        return text
    head = text[:text.index(begin) + len(begin)]
    tail = text[text.index(end):]
    return head + "\n" + body + "\n" + tail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CORRIDOR_REPORT.md's tables from the CSVs.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    data = {
        "eval": read(paths.EVALUATION_RAW),
        "robust": read(paths.ROBUSTNESS_RAW),
        "transfer": read(paths.TRANSFER_RAW),
    }

    built, skipped = [], []
    for name, builder in BUILDERS:
        body = builder(data)
        if body is None:
            skipped.append(name)
        else:
            built.append((name, body))

    for name, body in built:
        print("\n### %s\n" % name)
        print(body)
    if skipped:
        print("\n(no data yet for: %s)" % ", ".join(skipped))

    if args.write and built and REPORT_PATH.exists():
        text = REPORT_PATH.read_text(encoding="utf-8")
        for name, body in built:
            text = splice(text, name, body)
        REPORT_PATH.write_text(text, encoding="utf-8")
        print("\nUpdated %s (%s)"
              % (REPORT_PATH.name, ", ".join(n for n, _ in built)))


if __name__ == "__main__":
    main()
