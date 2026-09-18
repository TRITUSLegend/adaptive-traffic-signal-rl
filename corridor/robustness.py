"""
The robustness study: detector failure and zero-shot topology transfer.

Run:
    python corridor/robustness.py --detector      # the failure sweep
    python corridor/robustness.py --transfer      # the transfer study
    python corridor/robustness.py --all

Output:
    results/corridor/logs/corridor_robustness_raw.csv
    results/corridor/logs/corridor_transfer_raw.csv

WHY THE COMPARISON POINT IS FIXED-TIME'S OWN DEGRADATION
--------------------------------------------------------
The brief asks for robustness "measured against fixed-time control", and the
easy mistake is to report only how far the learned controller falls. That
number alone cannot distinguish a controller that degrades badly but still
beats fixed-time from one that degrades past it - and those are opposite
conclusions.

So every perturbed cell records all six controllers, and the reported quantity
is the MARGIN over fixed-time at each perturbation level, alongside each
controller's own degradation from its clean baseline.

FIXED-TIME'S IMMUNITY IS STRUCTURAL, NOT A VIRTUE
-------------------------------------------------
Fixed-time runs a clock and reads no detectors, so its detector-failure
degradation is zero **by construction** - the same kind of null Addendum E
documents for `fixed-time-tuned+safe` in the single-intersection study. It is
still run at every failure rate rather than assumed, because measuring it is
how we know the harness is doing what it claims, but the zero in that row is
arithmetic and the report says so.

The controllers that genuinely depend on detectors are actuated (gap-out needs
them), the greedy heuristic (it reads queues), and all three learned policies
(queues and densities are their observation).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sumo_setup  # noqa: E402,F401

from corridor import paths  # noqa: E402
from corridor import transfer as T  # noqa: E402
from corridor.env import run_corridor_episode  # noqa: E402
from corridor.evaluate_corridor import (  # noqa: E402
    CONTROLLER_ORDER,
    NEEDS_ACTUATED_NET,
    build_factory,
    load_policy,
)
from corridor.generate_corridor_routes import EVALUATION_SEEDS  # noqa: E402
from corridor.perturbations import make_perturbation  # noqa: E402

DEFAULT_JUNCTIONS = 3

# The scenario the failure sweep runs on. One scenario rather than all four,
# because the sweep multiplies quickly and the calibrated profile is the one
# the corridor was designed around.
FAILURE_SCENARIO = "calibrated"

# 0.0 is the clean control, measured through the identical code path so that a
# degradation is a difference between two rows of the same table.
FAILURE_RATES = [0.0, 0.125, 0.25, 0.5]
FAILURE_MODES = ["zero", "freeze"]

# Clean transfer targets: same junction structure, different topology. The only
# thing that changes is the corridor.
CLEAN_TARGETS = [2, 5]


def detector_sweep(controllers: list, rates: list, modes: list, seeds: list,
                   n_junctions: int) -> list:
    rows = []
    started = time.time()
    # The clean control is identical across modes, so it is run once per
    # controller rather than once per mode.
    jobs = [(c, r, m) for c in controllers for m in modes for r in rates
            if not (r == 0.0 and m != modes[0])]
    print("Detector-failure sweep: %d episodes" % (len(jobs) * len(seeds)))

    done = 0
    for controller, rate, mode in jobs:
        factory = build_factory(controller)
        actuated = controller in NEEDS_ACTUATED_NET
        for seed in seeds:
            perturbation = make_perturbation(rate, mode, FAILURE_SCENARIO, seed)
            result = run_corridor_episode(
                factory, controller, FAILURE_SCENARIO, seed,
                n_junctions=n_junctions, actuated=actuated,
                keep_trace=False, quiet=True, perturbation=perturbation)
            row = result.summary()
            # A clean row belongs to every mode when the table is read, so it
            # is labelled 'none' and matched on rate rather than on mode.
            rows.append(row)
            done += 1
            if done % 10 == 0:
                print("  %d/%d episodes, %.1f min elapsed"
                      % (done, len(jobs) * len(seeds),
                         (time.time() - started) / 60.0), flush=True)

    write_rows(paths.ROBUSTNESS_RAW, rows)
    print("Detector sweep finished in %.1f minutes."
          % ((time.time() - started) / 60.0))
    return rows


def transfer_factory(controller: str, target: str):
    """
    Build a controller for a network it did not train on.

    The baselines and the heuristic need no adaptation at all - SUMO drives the
    first two from the target's own program, and the heuristic reads whatever
    signal it finds. Only the learned policies are projected, and only on
    cologne3 does that projection actually do anything.
    """
    if controller in ("fixed-time", "actuated", "greedy-queue"):
        return build_factory(controller)
    if controller == "ma2c":
        return T.adapted_ma2c_factory(paths.model_file("ma2c"), "ma2c")
    return T.adapted_policy_factory(load_policy(controller), controller)


def clean_transfer(controllers: list, seeds: list) -> list:
    """Same junction structure, different corridor length. Nothing is adapted."""
    rows = []
    for n in CLEAN_TARGETS:
        net = paths.net_file(n)
        if not net.exists():
            print("  skipping corridor%d - not built "
                  "(python corridor/build_corridor.py --junctions %d)" % (n, n))
            continue
        for controller in controllers:
            factory = transfer_factory(controller, "corridor%d" % n)
            actuated = controller in NEEDS_ACTUATED_NET
            for seed in seeds:
                result = run_corridor_episode(
                    factory, controller, "calibrated", seed,
                    n_junctions=n, actuated=actuated, keep_trace=False,
                    quiet=True, network_label=paths.corridor_name(n))
                rows.append(result.summary())
            print("  corridor%d  %-13s done" % (n, controller), flush=True)
    return rows


def cologne_transfer(controllers: list, seeds: list) -> list:
    """
    Real-topology transfer onto a corridor we did not draw.

    Its junctions have 3/4/4 green phases and 5/6/8 lanes against the 4 and 8
    the policies trained on, so the learned controllers run through an explicit
    projection. The resulting degradation therefore mixes topology difference
    with adaptation cost, and is reported as such - the clean transfers above
    exist to give an unconfounded reading alongside it.
    """
    rows = []
    if not T.COLOGNE3_NET.exists():
        print("  skipping cologne3 - not found at %s" % T.COLOGNE3_NET)
        return rows

    actuated_net = T.cologne3_actuated_net()
    for controller in controllers:
        factory = transfer_factory(controller, "cologne3")
        net = actuated_net if controller in NEEDS_ACTUATED_NET else T.COLOGNE3_NET
        for seed in seeds:
            try:
                result = run_corridor_episode(
                    factory, controller, "cologne3", seed,
                    n_junctions=DEFAULT_JUNCTIONS, keep_trace=False, quiet=True,
                    net_override=net, route_override=T.COLOGNE3_ROUTES,
                    network_label="cologne3", begin_time=T.COLOGNE3_BEGIN)
            except Exception as exc:                      # noqa: BLE001
                # A transfer that cannot run is a legitimate finding, but it
                # must be visible rather than silently absent from the table.
                print("  cologne3 %-13s seed %d FAILED: %s"
                      % (controller, seed, exc), flush=True)
                continue
            rows.append(result.summary())
        print("  cologne3   %-13s done" % controller, flush=True)
    return rows


def write_rows(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corridor robustness study.")
    parser.add_argument("--detector", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--controllers", nargs="+", default=CONTROLLER_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=EVALUATION_SEEDS)
    parser.add_argument("--junctions", type=int, default=DEFAULT_JUNCTIONS)
    args = parser.parse_args()

    if not (args.detector or args.transfer or args.all):
        parser.error("choose --detector, --transfer or --all")

    if args.detector or args.all:
        detector_sweep(args.controllers, FAILURE_RATES, FAILURE_MODES,
                       args.seeds, args.junctions)
        print("  -> %s" % paths.ROBUSTNESS_RAW.relative_to(paths.ROOT).as_posix())

    if args.transfer or args.all:
        started = time.time()
        print("Topology transfer:")
        rows = clean_transfer(args.controllers, args.seeds)
        rows += cologne_transfer(args.controllers, args.seeds)
        write_rows(paths.TRANSFER_RAW, rows)
        print("Transfer finished in %.1f minutes."
              % ((time.time() - started) / 60.0))
        print("  -> %s" % paths.TRANSFER_RAW.relative_to(paths.ROOT).as_posix())


if __name__ == "__main__":
    main()
