"""
Evaluation harness: run every controller on every scenario and tabulate the result.

Run:
    python eval/evaluate.py                          # everything (~26 min)
    python eval/evaluate.py --scenarios ns_heavy     # one scenario
    python eval/evaluate.py --controllers fixed-time dqn
    python eval/evaluate.py --seeds 100 101          # fewer repeats, faster

Outputs:
    results/logs/evaluation_raw.csv       one row per simulated hour
    results/logs/evaluation_summary.csv   mean and spread per controller x scenario
    results/logs/trace_*.csv              second-by-second traces for the plots

HOW THE COMPARISON IS KEPT FAIR
-------------------------------
Every controller is run through the same `run_episode` function, on the same
network, with the same amber time, the same minimum green, the same episode
length, and - crucially - the *same vehicles*. Scenario and seed together pick
a route file that lists exactly which car arrives when and where it is going,
so when we say the DQN beat the fixed-time plan, both of them were driving the
identical hour of traffic.

WHY FIVE SEEDS AND NOT ONE
--------------------------
Traffic is random. Two hours with the same average demand can differ by several
seconds of mean waiting time purely by chance, which is easily large enough to
manufacture a result that is not there. We therefore run five independently
generated hours per scenario and report the mean along with the spread, so a
difference can be judged against the noise it has to beat.

None of these five seeds were used during training.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline.fixed_time import tuned_fixed_time_factory  # noqa: E402
from eval.simulation import (  # noqa: E402
    fixed_time_factory,
    greedy_factory,
    policy_factory,
    run_episode,
    safety_wrapped_factory,
)
from eval.paths import LOGS_DIR, trace_path  # noqa: E402
from networks.generate_routes import EVALUATION_SEEDS, SCENARIOS  # noqa: E402

MODELS_DIR = ROOT / "agent" / "models"

# Controllers in the order they should appear in tables and plots: the two
# non-adaptive baselines first, then the heuristic, then the learned agents.
CONTROLLER_ORDER = ["fixed-time", "fixed-time-tuned", "greedy-queue", "dqn", "ppo"]

# The metrics we report, with the direction that counts as "better".
METRICS = [
    ("mean_waiting_time_s", "mean wait (s)", "lower"),
    ("mean_queue_length_veh", "mean queue (veh)", "lower"),
    ("mean_time_loss_s", "time lost (s)", "lower"),
    ("throughput_veh", "throughput (veh)", "higher"),
]


def load_policy(name: str):
    """
    Load a trained stable-baselines3 model, or return None if not trained yet.

    `name` is the model's filename stem, which is also the name it is reported
    under: "dqn" and "ppo" are the two the default sweep uses, and any variant
    trained with `--tag` ("dqn_seed1", "ppo_long", ...) loads here too. The
    algorithm is taken from the leading word, because that is what the training
    scripts put there and a PPO archive cannot be opened by DQN.load.
    """
    path = MODELS_DIR / (name + ".zip")
    if not path.exists():
        return None
    algo = name.split("_")[0]
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(str(path), device="cpu")
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(str(path), device="cpu")
    raise ValueError(
        "cannot tell which algorithm %r is: a model name must begin with "
        "'dqn' or 'ppo'" % name)


def is_learned(controller: str) -> bool:
    """True for anything loaded from agent/models/, including tagged variants."""
    base = controller[:-len("+safe")] if controller.endswith("+safe") else controller
    return base.split("_")[0] in ("dqn", "ppo")


def model_name(controller: str) -> str:
    """The model file a controller name refers to, ignoring any +safe suffix."""
    return controller[:-len("+safe")] if controller.endswith("+safe") else controller


def build_factory(controller: str, scenario: str):
    """
    Make the controller factory for one (controller, scenario) pair.

    The scenario matters only for `fixed-time-tuned`, whose whole point is that
    its timings are calculated from the demand it is about to face.

    A `+safe` suffix wraps the controller in the maximum-green and
    anti-starvation bounds (see SafetyConstrainedController). Written as a
    suffix rather than as a separate controller so that any policy can be
    measured with and without the constraint, on the same code path - which is
    what turns "the constraint is satisfied" into a cost in seconds.
    """
    if controller.endswith("+safe"):
        inner = build_factory(controller[:-len("+safe")], scenario)
        return None if inner is None else safety_wrapped_factory(inner)

    if controller == "fixed-time":
        return fixed_time_factory()
    if controller == "fixed-time-tuned":
        return tuned_fixed_time_factory(scenario)
    if controller == "greedy-queue":
        return greedy_factory()
    if is_learned(controller):
        model = load_policy(controller)
        if model is None:
            return None
        return policy_factory(model, controller)
    raise ValueError("unknown controller: " + controller)


def write_trace(result, path: Path) -> None:
    """Save the per-step trace of one episode, used for the time-series plot."""
    if not result.trace:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result.trace[0].keys()))
        writer.writeheader()
        writer.writerows(result.trace)


def merge_with_existing(new_rows: list, raw_path: Path) -> list:
    """
    Combine freshly measured episodes with the ones already on disk.

    An episode is identified by (controller, scenario, seed). If the new run
    repeats one that already exists, the NEW result wins - re-running a
    controller is how you replace a stale measurement, so silently keeping the
    old number would be the wrong way round.

    Numbers read back from CSV are strings, so they are converted to the types
    the summary maths expects.
    """
    def key(row):
        return (row["controller"], row["scenario"], int(row["seed"]))

    merged = {}
    with raw_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            typed = dict(row)
            for field, value in typed.items():
                if field in ("controller", "scenario"):
                    continue
                try:
                    typed[field] = float(value)
                except (TypeError, ValueError):
                    pass
            typed["seed"] = int(typed["seed"])
            merged[key(typed)] = typed

    replaced = 0
    for row in new_rows:
        if key(row) in merged:
            replaced += 1
        merged[key(row)] = row

    print("\nMerged into %s: %d existing episodes kept, %d replaced, %d added."
          % (raw_path.name, len(merged) - len(new_rows),
             replaced, len(new_rows) - replaced))
    return list(merged.values())


def summarise(rows: list) -> list:
    """Collapse the per-episode rows into one row per controller x scenario."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["controller"], row["scenario"]), []).append(row)

    summary = []
    for (controller, scenario), group in grouped.items():
        entry = {
            "controller": controller,
            "scenario": scenario,
            "n_runs": len(group),
        }
        for key, _, _ in METRICS:
            values = [r[key] for r in group]
            entry[key + "_mean"] = round(statistics.fmean(values), 2)
            # Standard deviation across seeds: how much this number moves from
            # one random hour to the next. A gap between controllers is only
            # meaningful if it is comfortably bigger than this.
            entry[key + "_std"] = round(
                statistics.stdev(values) if len(values) > 1 else 0.0, 2)
        entry["phase_switches_mean"] = round(
            statistics.fmean([r["phase_switches"] for r in group]), 1)
        entry["vehicles_not_inserted_max"] = max(
            r["vehicles_not_inserted"] for r in group)
        summary.append(entry)

    order = {c: i for i, c in enumerate(CONTROLLER_ORDER)}
    scenarios = list(SCENARIOS)
    summary.sort(key=lambda e: (scenarios.index(e["scenario"]),
                                order.get(e["controller"], 99)))
    return summary


def print_table(summary: list) -> None:
    """Print the headline comparison, one block per scenario."""
    scenarios = []
    for entry in summary:
        if entry["scenario"] not in scenarios:
            scenarios.append(entry["scenario"])

    for scenario in scenarios:
        rows = [e for e in summary if e["scenario"] == scenario]
        note = SCENARIOS[scenario]["note"].split(".")[0]
        print("\n" + "=" * 88)
        print("%s  -  %s" % (scenario.upper(), note))
        print("=" * 88)
        print("%-17s %13s %13s %13s %13s %8s"
              % ("controller", "wait (s)", "queue (veh)", "time lost (s)",
                 "throughput", "switches"))
        print("-" * 88)

        baseline = next((r for r in rows if r["controller"] == "fixed-time"), None)
        for row in rows:
            cells = []
            for key, _, _ in METRICS:
                cells.append("%8.1f+-%-4.1f" % (row[key + "_mean"], row[key + "_std"]))
            print("%-17s %13s %13s %13s %13s %8.0f"
                  % (row["controller"], cells[0], cells[1], cells[2], cells[3],
                     row["phase_switches_mean"]))

        if baseline is not None:
            print("-" * 88)
            print("change vs fixed-time baseline:")
            for row in rows:
                if row["controller"] == "fixed-time":
                    continue
                parts = []
                for key, label, better in METRICS:
                    base = baseline[key + "_mean"]
                    value = row[key + "_mean"]
                    if base == 0:
                        continue
                    pct = 100.0 * (value - base) / base
                    good = (pct < 0) if better == "lower" else (pct > 0)
                    parts.append("%s %+.1f%%%s" % (label, pct, "" if good else " (worse)"))
                print("  %-17s %s" % (row["controller"], "   ".join(parts)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare traffic signal controllers.")
    parser.add_argument("--controllers", nargs="+", default=None,
                        help="subset of: " + " ".join(CONTROLLER_ORDER) +
                             "; a tagged model such as dqn_seed1 or ppo_long "
                             "also works if agent/models/<name>.zip exists")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        help="subset of: " + " ".join(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=EVALUATION_SEEDS,
                        help="evaluation seeds (default: %s)" % EVALUATION_SEEDS)
    parser.add_argument("--trace-seed", type=int, default=EVALUATION_SEEDS[0],
                        help="seed whose per-step trace is saved for plotting")
    parser.add_argument("--append", action="store_true",
                        help="merge these results into the existing "
                             "evaluation_raw.csv instead of replacing it, so a "
                             "newly trained controller can be added without "
                             "re-running the whole sweep")
    args = parser.parse_args()

    controllers = args.controllers or CONTROLLER_ORDER
    # Silently drop learned controllers that have not been trained yet, so the
    # harness still works before/without a PPO run.
    available = []
    for name in controllers:
        if is_learned(name) and load_policy(model_name(name)) is None:
            print("skipping %s - no trained model at agent/models/%s.zip"
                  % (name, name))
            continue
        available.append(name)
    if not available:
        raise SystemExit("no controllers to evaluate")

    total = len(available) * len(args.scenarios) * len(args.seeds)
    print("Evaluating %d controller(s) x %d scenario(s) x %d seed(s) = %d hours "
          "of simulated traffic.\n" % (len(available), len(args.scenarios),
                                       len(args.seeds), total))

    rows = []
    started = time.time()
    done = 0
    for scenario in args.scenarios:
        print("\n--- %s ---" % scenario)
        for controller in available:
            factory = build_factory(controller, scenario)
            episode_values = []
            for seed in args.seeds:
                result = run_episode(factory, controller, scenario, seed,
                                     keep_trace=(seed == args.trace_seed))
                rows.append(result.summary())
                episode_values.append(result.mean_waiting_time_s)
                if seed == args.trace_seed:
                    write_trace(result, trace_path(controller, scenario))
                done += 1
            print("  %-17s wait %6.1f s  (%d runs, %.0f%% done, %.1f min elapsed)"
                  % (controller, statistics.fmean(episode_values), len(episode_values),
                     100.0 * done / total, (time.time() - started) / 60.0))

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = LOGS_DIR / "evaluation_raw.csv"

    if args.append and raw_path.exists():
        rows = merge_with_existing(rows, raw_path)

    with raw_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)
    summary_path = LOGS_DIR / "evaluation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print_table(summary)

    # The honesty check described in eval/simulation.py: if SUMO could not
    # insert the demanded vehicles, congestion is being hidden outside the
    # simulation and the numbers above understate it.
    worst = max(e["vehicles_not_inserted_max"] for e in summary)
    unmeasured = [e for e in summary if e["vehicles_not_inserted_max"] < 0]
    print("\n" + "=" * 88)
    if unmeasured:
        # -1 is the sentinel run_episode stores when SUMO could not be asked.
        # It must not be read as "zero vehicles were dropped".
        print("WARNING: the insertion backlog could not be measured for %d "
              "controller/scenario pair(s):" % len(unmeasured))
        for entry in unmeasured:
            print("    %s on %s" % (entry["controller"], entry["scenario"]))
        print("These are recorded as -1, NOT as zero. The honesty check below "
              "cannot be relied on until they are re-run.")
    print("Worst-case vehicles still awaiting insertion at the hour's end: %d "
          "(near zero means no demand was silently dropped)" % worst)
    print("Wrote %s" % raw_path)
    print("Wrote %s" % summary_path)
    print("Total time: %.1f minutes" % ((time.time() - started) / 60.0))


if __name__ == "__main__":
    main()
