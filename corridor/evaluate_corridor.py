"""
Evaluate every corridor controller on every scenario, through one code path.

Run:
    python corridor/evaluate_corridor.py                    # the full sweep
    python corridor/evaluate_corridor.py --scenarios calibrated
    python corridor/evaluate_corridor.py --controllers fixed-time actuated

Output:
    results/corridor/logs/corridor_evaluation_raw.csv      one row per episode
    results/corridor/logs/corridor_evaluation_summary.csv  mean/sd per cell
    results/corridor/logs/trace_<controller>_<scenario>_corridor3.csv

WHAT MAKES THIS A FAIR COMPARISON
---------------------------------
The same four guarantees the single-intersection study relies on, restated
because they are the whole value of the numbers:

  * **Same code path.** Fixed-time, actuated, the heuristic and all three
    learned controllers run through `run_corridor_episode`. They are not merely
    measured with the same metrics, they are measured by the same code.
  * **Same vehicles.** A (scenario, seed) pair names a file listing exactly
    which car arrives when and where it is going. Every controller drives the
    identical hour.
  * **Same safety constraints.** 4 s amber and 10 s minimum green for
    everything, including the actuated plan, whose minDur/maxDur are pinned to
    the project's own bounds rather than to SUMO's defaults.
  * **Unseen demand.** Training used seeds 0-9; every number here is measured
    on seeds 100-104.

THROUGHPUT IS ALWAYS CARRIED ALONGSIDE WAITING TIME
---------------------------------------------------
A controller can make waiting time look excellent by refusing to admit
vehicles, so every row records throughput and the insertion backlog next to the
delay. On a corridor this matters more than at a single junction, because a
signal can also protect its own numbers by starving the link that feeds its
neighbour.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sumo_setup  # noqa: E402,F401

from corridor import controllers as C  # noqa: E402
from corridor import paths  # noqa: E402
from corridor.env import run_corridor_episode  # noqa: E402
from corridor.generate_corridor_routes import (  # noqa: E402
    EVALUATION_SEEDS,
    SCENARIOS,
)

DEFAULT_JUNCTIONS = 3

# The controllers the headline table describes, in reporting order: the two
# required baselines, the non-learned reference, then the three learned ones.
CONTROLLER_ORDER = [
    "fixed-time",
    "actuated",
    "greedy-queue",
    "idqn",
    "ippo",
    "ma2c",
]

LABELS = {
    "fixed-time": "Fixed-time",
    "actuated": "Actuated (gap-based)",
    "greedy-queue": "Greedy queue heuristic",
    "idqn": "Independent DQN",
    "ippo": "Independent PPO",
    "ma2c": "MA2C",
}

# Which controllers SUMO drives itself, and therefore which need the actuated
# network rather than the static one.
NEEDS_ACTUATED_NET = {"actuated"}

METRICS = [
    "mean_waiting_time_s",
    "mean_travel_time_s",
    "mean_time_loss_s",
    "mean_queue_length_veh",
    "max_queue_length_veh",
    "throughput_veh",
    "vehicles_entered",
    "vehicles_not_inserted",
    "phase_switches",
    "total_reward",
]


def load_policy(name: str):
    """Load a trained corridor policy, or explain what is missing."""
    from stable_baselines3 import DQN, PPO

    path = paths.model_file(name)
    if not path.exists():
        raise FileNotFoundError(
            "%s is missing - train it with:  python corridor/%s"
            % (path, "train_ma2c.py" if name == "ma2c"
               else "train_independent.py --algo %s" % name[1:]))
    if name == "ma2c":
        from corridor.ma2c import MA2C
        return MA2C.load(path)
    if name.endswith("dqn"):
        return DQN.load(path)
    if name.endswith("ppo"):
        return PPO.load(path)
    raise ValueError("no loader for controller %r" % name)


def build_factory(name: str):
    """Map a controller name onto the factory `run_corridor_episode` wants."""
    if name == "fixed-time":
        return C.fixed_time_factory()
    if name == "actuated":
        return C.actuated_factory()
    if name == "greedy-queue":
        return C.greedy_factory()

    base = name[:-5] if name.endswith("+safe") else name
    if base == "ma2c":
        from corridor.ma2c import ma2c_factory
        inner = ma2c_factory(load_policy("ma2c"), base)
    else:
        inner = C.policy_factory(load_policy(base), base)

    if name.endswith("+safe"):
        return C.safety_wrapped_factory(inner)
    return inner


def evaluate(controllers: list, scenarios: list, seeds: list,
             n_junctions: int, append: bool = False, quiet: bool = False) -> list:
    rows = []
    started = time.time()
    total = len(controllers) * len(scenarios) * len(seeds)
    done = 0

    for controller in controllers:
        factory = build_factory(controller)
        actuated = controller in NEEDS_ACTUATED_NET
        for scenario in scenarios:
            traces = []
            for seed in seeds:
                result = run_corridor_episode(
                    factory, controller, scenario, seed,
                    n_junctions=n_junctions, actuated=actuated,
                    keep_trace=True, quiet=quiet)
                rows.append(result.summary())
                if not traces:
                    traces = result.trace
                done += 1
                if quiet and done % 5 == 0:
                    print("  %d/%d episodes, %.1f min elapsed"
                          % (done, total, (time.time() - started) / 60.0),
                          flush=True)
            if traces:
                write_trace(controller, scenario, traces, n_junctions)

    write_raw(rows, append=append)
    write_summary()
    print("Evaluated %d episodes in %.1f minutes."
          % (len(rows), (time.time() - started) / 60.0))
    return rows


def write_trace(controller: str, scenario: str, trace: list,
                n_junctions: int) -> None:
    path = paths.trace_file(controller, scenario, n_junctions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)


def read_raw() -> list:
    if not paths.EVALUATION_RAW.exists():
        return []
    with paths.EVALUATION_RAW.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_raw(rows: list, append: bool = False) -> None:
    """
    Write the per-episode rows, keyed by (controller, scenario, seed).

    In append mode a re-run REPLACES its row rather than adding a duplicate,
    which is what makes the sweep safe to re-run for one controller without
    re-running everything - the same contract `eval/evaluate.py` offers.
    """
    paths.EVALUATION_RAW.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if append:
        for row in read_raw():
            merged[(row["controller"], row["scenario"], row["seed"])] = row
    for row in rows:
        merged[(row["controller"], row["scenario"], str(row["seed"]))] = row

    ordered = sorted(merged.values(),
                     key=lambda r: (str(r["controller"]), str(r["scenario"]),
                                    int(r["seed"])))
    with paths.EVALUATION_RAW.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def write_summary() -> None:
    """Mean and standard deviation per (controller, scenario) cell."""
    rows = read_raw()
    if not rows:
        return
    cells = {}
    for row in rows:
        cells.setdefault((row["controller"], row["scenario"]), []).append(row)

    out = []
    for (controller, scenario), group in sorted(cells.items()):
        entry = {"controller": controller, "scenario": scenario,
                 "n_runs": len(group)}
        for metric in METRICS:
            values = [float(r[metric]) for r in group if r.get(metric) not in
                      (None, "")]
            if not values:
                continue
            entry[metric + "_mean"] = round(statistics.mean(values), 3)
            entry[metric + "_std"] = round(
                statistics.stdev(values) if len(values) > 1 else 0.0, 3)
        out.append(entry)

    with paths.EVALUATION_SUMMARY.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every corridor controller on every scenario.")
    parser.add_argument("--controllers", nargs="+", default=CONTROLLER_ORDER)
    parser.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=EVALUATION_SEEDS)
    parser.add_argument("--junctions", type=int, default=DEFAULT_JUNCTIONS)
    parser.add_argument("--append", action="store_true",
                        help="merge into the existing raw CSV instead of "
                             "replacing it")
    args = parser.parse_args()

    print("Evaluating %d controllers x %d scenarios x %d seeds"
          % (len(args.controllers), len(args.scenarios), len(args.seeds)))
    evaluate(args.controllers, args.scenarios, args.seeds, args.junctions,
             append=args.append, quiet=True)
    print("  raw     -> %s" % paths.EVALUATION_RAW.relative_to(paths.ROOT).as_posix())
    print("  summary -> %s"
          % paths.EVALUATION_SUMMARY.relative_to(paths.ROOT).as_posix())


if __name__ == "__main__":
    main()
