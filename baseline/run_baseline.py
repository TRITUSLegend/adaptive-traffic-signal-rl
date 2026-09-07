"""
Run the fixed-time baseline controllers and report their metrics.

Run:
    python baseline/run_baseline.py                       # all scenarios, 5 seeds
    python baseline/run_baseline.py --scenarios ns_heavy
    python baseline/run_baseline.py --seeds 100 101       # quicker
    python baseline/run_baseline.py --untuned-only

Outputs:
    results/logs/baseline_raw.csv       one row per simulated hour
    results/logs/baseline_summary.csv   mean and spread per controller x scenario

WHY THIS SCRIPT EXISTS SEPARATELY
---------------------------------
`eval/evaluate.py` also runs these controllers, as part of the full comparison.
This script exists so the baseline can be produced and inspected on its own,
before any agent has been trained - which is the order the work actually
happens in. You cannot tell whether a learned controller is any good until you
know what the ordinary one does, and you want that number in hand early, while
there is still time to fix the simulation if it looks wrong.

WHAT "FIXED-TIME" MEANS
-----------------------
A fixed-time signal runs a timing plan decided in advance and never looks at
traffic. Each phase gets its allotted seconds whether thirty cars are waiting
or none. This is how the large majority of real signalised intersections
actually operate, which is what makes it the right thing to compare against.

Two versions are measured:

  fixed-time         Equal green for north-south and east-west: a plan nobody
                     has retimed for current demand.
  fixed-time-tuned   Green split computed from the demand with Webster's
                     method - what a traffic engineer would install.
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

from baseline.fixed_time import describe_plan, tuned_fixed_time_factory  # noqa: E402
from eval.simulation import fixed_time_factory, run_episode  # noqa: E402
from networks.generate_routes import EVALUATION_SEEDS, SCENARIOS  # noqa: E402

LOGS_DIR = ROOT / "results" / "logs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-time baselines.")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        help="subset of: " + " ".join(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=EVALUATION_SEEDS,
                        help="evaluation seeds (default: %s)" % EVALUATION_SEEDS)
    parser.add_argument("--untuned-only", action="store_true",
                        help="skip the Webster-tuned plan")
    args = parser.parse_args()

    controllers = ["fixed-time"] if args.untuned_only else [
        "fixed-time", "fixed-time-tuned"]

    total = len(controllers) * len(args.scenarios) * len(args.seeds)
    print("Fixed-time baseline: %d controller(s) x %d scenario(s) x %d seed(s) "
          "= %d simulated hours.\n" % (len(controllers), len(args.scenarios),
                                       len(args.seeds), total))

    rows = []
    started = time.time()

    for scenario in args.scenarios:
        print("\n" + "=" * 76)
        print("%s  -  %s" % (scenario.upper(), SCENARIOS[scenario]["note"]))
        flows = SCENARIOS[scenario]["flows"]
        print("demand: " + ", ".join("%s %d veh/h" % (a, v) for a, v in flows.items()))
        if not args.untuned_only:
            print(describe_plan(scenario))
        print("=" * 76)

        for controller in controllers:
            factory = (fixed_time_factory() if controller == "fixed-time"
                       else tuned_fixed_time_factory(scenario))
            episodes = []
            for seed in args.seeds:
                result = run_episode(factory, controller, scenario, seed,
                                     keep_trace=False)
                rows.append(result.summary())
                episodes.append(result)

            waits = [r.mean_waiting_time_s for r in episodes]
            queues = [r.mean_queue_length_veh for r in episodes]
            through = [r.throughput_veh for r in episodes]
            spread = statistics.stdev(waits) if len(waits) > 1 else 0.0
            print("  %-17s wait %6.1f s (+-%4.1f)  queue %5.1f veh  "
                  "throughput %4.0f veh"
                  % (controller, statistics.fmean(waits), spread,
                     statistics.fmean(queues), statistics.fmean(through)))

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = LOGS_DIR / "baseline_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Collapse to one row per controller x scenario.
    grouped = {}
    for row in rows:
        grouped.setdefault((row["controller"], row["scenario"]), []).append(row)
    summary = []
    for (controller, scenario), group in grouped.items():
        entry = {"controller": controller, "scenario": scenario,
                 "n_runs": len(group)}
        for field in ("mean_waiting_time_s", "mean_queue_length_veh",
                      "mean_time_loss_s", "throughput_veh"):
            values = [r[field] for r in group]
            entry[field + "_mean"] = round(statistics.fmean(values), 2)
            entry[field + "_std"] = round(
                statistics.stdev(values) if len(values) > 1 else 0.0, 2)
        summary.append(entry)

    summary_path = LOGS_DIR / "baseline_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nWrote %s" % raw_path)
    print("Wrote %s" % summary_path)
    print("Total time: %.1f minutes" % ((time.time() - started) / 60.0))
    print("\nNext: train an agent with  python agent/train_dqn.py")


if __name__ == "__main__":
    main()
