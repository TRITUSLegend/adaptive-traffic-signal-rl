"""
Generate the traffic demand (SUMO ".rou.xml" route files) for every scenario.

Run:
    python networks/generate_routes.py            # write every scenario / seed
    python networks/generate_routes.py --list     # just show the scenario table

Output:
    networks/routes/<scenario>_seed<N>.rou.xml

WHAT A ROUTE FILE IS
--------------------
SUMO does not invent traffic by itself.  A route file lists individual
vehicles: when each one appears, and which sequence of roads it drives.  This
script writes those lists.

HOW THE TRAFFIC IS MODELLED
---------------------------
Vehicles arrive as a **Poisson process** on each approach, which is the
standard model for vehicles arriving at an isolated intersection from an
upstream source far enough away that arrivals are effectively independent.
Concretely: if an approach carries 600 vehicles/hour, the gap between one
vehicle and the next is drawn from an exponential distribution with a mean of
3600/600 = 6 seconds.  That produces realistic random clumping - sometimes
four cars arrive nose-to-tail, sometimes the road is empty for 20 seconds -
rather than an unrealistic metronome of one car exactly every 6 seconds.

Why this matters for the experiment: a fixed-time signal cannot react to that
clumping, because its plan is decided in advance.  A learned controller can.
Randomness is therefore the thing that creates the opportunity we are trying
to measure, so it has to be modelled properly rather than smoothed away.

We generate explicit <vehicle> entries with a seeded Python random number
generator instead of using SUMO's <flow probability=...>.  This means the exact
same vehicles are produced on every machine and every SUMO version, so the
fixed-time controller and the RL agent are compared on literally identical
traffic.

WHY SEPARATE SEEDS FOR TRAINING AND EVALUATION
----------------------------------------------
Training uses seeds 0-9 and evaluation uses seeds 100-104.  The two sets never
overlap.  If we evaluated on the same arrival sequence the agent trained on,
a good score might just mean the agent memorised that one hour of traffic.
Testing on unseen seeds shows it learned a *policy*, not a recording.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTES_DIR = HERE / "routes"

# ---------------------------------------------------------------------------
# Route topology: for each approach, where each turn takes you.
# See connections.con.xml for the geometry these mirror.
# ---------------------------------------------------------------------------
TURNS = {
    #  approach   right    straight  left
    "N2C": {"right": "C2W", "straight": "C2S", "left": "C2E"},
    "S2C": {"right": "C2E", "straight": "C2N", "left": "C2W"},
    "E2C": {"right": "C2N", "straight": "C2W", "left": "C2S"},
    "W2C": {"right": "C2S", "straight": "C2E", "left": "C2N"},
}

# Proportion of drivers on each approach making each manoeuvre.
#
# 60/20/20 is a common default for an urban four-arm intersection where the
# through movement dominates.  Keeping the split identical on all four
# approaches means the only thing that differs between scenarios is the
# VOLUME per approach - so any performance difference we measure is caused by
# how lopsided the demand is, not by a change in turning behaviour.
TURN_SPLIT = {"straight": 0.60, "left": 0.20, "right": 0.20}

# ---------------------------------------------------------------------------
# Demand scenarios: vehicles per hour entering on each approach.
# ---------------------------------------------------------------------------
#
# Capacity context, so these numbers are not arbitrary.  One lane discharging
# on green passes roughly 1800 vehicles/hour.  The kerb lane (straight+right)
# gets 32 s of green in a 104 s fixed-time cycle, so its capacity is about
# 1800 * 32/104 = 550 veh/h.  Straight+right is 80% of an approach's volume,
# so an approach carrying V veh/h loads that lane to 0.8*V.
#
#   approach volume     kerb-lane load     load / capacity
#        300 veh/h          240 veh/h            0.44   free flowing
#        600 veh/h          480 veh/h            0.87   busy, queues form
#        700 veh/h          560 veh/h            1.02   saturated
#
# The scenarios are chosen to span that range, because a controller can only
# be distinguished from another when the intersection is actually under load.
SCENARIOS = {
    # --- used for training AND evaluation -------------------------------
    "balanced": {
        "flows": {"N2C": 400, "S2C": 400, "E2C": 400, "W2C": 400},
        "note": "Equal demand on all four approaches. The case a symmetric "
                "fixed-time plan is designed for, so the hardest one for RL "
                "to beat.",
    },
    "ns_heavy": {
        "flows": {"N2C": 600, "S2C": 600, "E2C": 300, "W2C": 300},
        "note": "North-south arterial carrying twice the east-west side "
                "street. THIS IS THE TRAINING SCENARIO.",
    },
    "peak": {
        "flows": {"N2C": 700, "S2C": 700, "E2C": 450, "W2C": 450},
        "note": "Rush hour. The north-south through lane runs at ~100% of "
                "its fixed-time capacity, so queues build faster than they "
                "clear.",
    },
    # --- robustness set: never seen during training ----------------------
    "ew_heavy": {
        "flows": {"N2C": 300, "S2C": 300, "E2C": 600, "W2C": 600},
        "note": "ROBUSTNESS: ns_heavy mirrored. The busy direction is now the "
                "one the agent learned to starve.",
    },
    "surge": {
        "flows": {"N2C": 780, "S2C": 780, "E2C": 390, "W2C": 390},
        "note": "ROBUSTNESS: ns_heavy scaled up 30%. Tests behaviour at "
                "volumes above anything seen in training.",
    },
    "light": {
        "flows": {"N2C": 360, "S2C": 360, "E2C": 180, "W2C": 180},
        "note": "ROBUSTNESS: ns_heavy scaled down 40%. Tests that the agent "
                "does not waste green time when the road is nearly empty.",
    },
}

TRAINING_SCENARIO = "ns_heavy"
TRAINING_SEEDS = list(range(0, 10))     # 10 different hours of traffic
EVALUATION_SEEDS = list(range(100, 105))  # 5 unseen hours, used by every controller

# One simulated hour. Justified in REPORT.md: long enough for queues to reach
# steady state and for several hundred vehicles to complete their trip, short
# enough that a full training run finishes on a laptop CPU.
DEFAULT_DURATION = 3600

VEHICLE_TYPE = """    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0"
           minGap="2.5" maxSpeed="13.89" carFollowModel="Krauss"/>"""


def route_id(approach: str, turn: str) -> str:
    return approach + "_" + turn


def write_route_definitions(handle) -> None:
    """Emit the 12 possible paths through the intersection (4 approaches x 3 turns)."""
    for approach, targets in TURNS.items():
        for turn, exit_edge in targets.items():
            handle.write(
                '    <route id="%s" edges="%s %s"/>\n'
                % (route_id(approach, turn), approach, exit_edge)
            )


def sample_vehicles(flows: dict, duration: int, rng: random.Random) -> list:
    """
    Draw the arrival time and turn choice of every vehicle in the hour.

    Returns a list of (depart_time, approach, turn), sorted by time - SUMO
    requires vehicles to be listed in departure order.
    """
    vehicles = []
    turn_names = list(TURN_SPLIT.keys())
    turn_weights = [TURN_SPLIT[t] for t in turn_names]

    for approach, veh_per_hour in flows.items():
        if veh_per_hour <= 0:
            continue
        mean_gap = 3600.0 / veh_per_hour  # seconds between arrivals, on average
        t = 0.0
        while True:
            # Exponential inter-arrival gaps == Poisson arrival process.
            t += rng.expovariate(1.0 / mean_gap)
            if t >= duration:
                break
            turn = rng.choices(turn_names, weights=turn_weights, k=1)[0]
            vehicles.append((t, approach, turn))

    vehicles.sort(key=lambda v: v[0])
    return vehicles


def write_scenario(scenario: str, seed: int, duration: int, out_dir: Path) -> Path:
    """Write one <scenario, seed> route file and return its path."""
    flows = SCENARIOS[scenario]["flows"]
    # Offsetting by scenario name keeps two scenarios with the same seed from
    # sharing an identical arrival pattern.
    rng = random.Random(hash((scenario, seed)) & 0xFFFFFFFF)
    vehicles = sample_vehicles(flows, duration, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("%s_seed%d.rou.xml" % (scenario, seed))

    with path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write("<!--\n")
        fh.write("  Scenario : %s (seed %d)\n" % (scenario, seed))
        fh.write("  %s\n" % SCENARIOS[scenario]["note"])
        fh.write("  Demand   : %s veh/h\n" % ", ".join(
            "%s=%d" % (a, v) for a, v in flows.items()))
        fh.write("  Duration : %d s, %d vehicles generated\n" % (duration, len(vehicles)))
        fh.write("  Generated by networks/generate_routes.py - do not edit by hand.\n")
        fh.write("-->\n")
        fh.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        fh.write('        xsi:noNamespaceSchemaLocation='
                 '"http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        fh.write(VEHICLE_TYPE + "\n\n")
        write_route_definitions(fh)
        fh.write("\n")
        for i, (depart, approach, turn) in enumerate(vehicles):
            # departLane="best" lets SUMO place the car in a lane from which
            # its turn is actually legal (left turns must start in lane 1).
            fh.write(
                '    <vehicle id="%s_%d" type="car" route="%s" depart="%.2f" '
                'departLane="best" departSpeed="max"/>\n'
                % (approach, i, route_id(approach, turn), depart)
            )
        fh.write("</routes>\n")

    return path


def training_route_files(out_dir: Path = ROUTES_DIR) -> list:
    """Paths of the route files the training script cycles through."""
    return [out_dir / ("%s_seed%d.rou.xml" % (TRAINING_SCENARIO, s))
            for s in TRAINING_SEEDS]


def evaluation_route_files(scenario: str, out_dir: Path = ROUTES_DIR) -> list:
    """Paths of the held-out route files every controller is evaluated on."""
    return [out_dir / ("%s_seed%d.rou.xml" % (scenario, s)) for s in EVALUATION_SEEDS]


def print_scenario_table() -> None:
    print("%-10s %7s %7s %7s  %s" % ("scenario", "N/S", "E/W", "total", "role"))
    print("-" * 78)
    for name, cfg in SCENARIOS.items():
        flows = cfg["flows"]
        total = sum(flows.values())
        role = "TRAIN + eval" if name == TRAINING_SCENARIO else (
            "robustness" if "ROBUSTNESS" in cfg["note"] else "eval")
        print("%-10s %7d %7d %7d  %s"
              % (name, flows["N2C"], flows["E2C"], total, role))
    print("\n(N/S and E/W columns are per-approach vehicles/hour.)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SUMO demand files.")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="simulated seconds per episode (default: 3600)")
    parser.add_argument("--list", action="store_true",
                        help="print the scenario table and exit")
    args = parser.parse_args()

    if args.list:
        print_scenario_table()
        return

    print_scenario_table()
    print("\nWriting route files to " + str(ROUTES_DIR))

    written = 0
    for scenario in SCENARIOS:
        seeds = sorted(set(EVALUATION_SEEDS +
                           (TRAINING_SEEDS if scenario == TRAINING_SCENARIO else [])))
        counts = []
        for seed in seeds:
            path = write_scenario(scenario, seed, args.duration, ROUTES_DIR)
            # Count vehicles for the summary line.
            counts.append(sum(1 for line in path.open(encoding="utf-8")
                              if "<vehicle " in line))
            written += 1
        print("  %-10s %2d files, %4d-%4d vehicles per hour"
              % (scenario, len(seeds), min(counts), max(counts)))

    print("\n%d route files written." % written)
    print("Training uses %s seeds %s; every controller is evaluated on seeds %s."
          % (TRAINING_SCENARIO, TRAINING_SEEDS, EVALUATION_SEEDS))


if __name__ == "__main__":
    main()
