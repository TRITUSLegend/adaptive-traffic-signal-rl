"""
Generate Poisson vehicle demand for the corridor, calibrated from real data.

Run:
    python corridor/generate_corridor_routes.py            # every scenario/seed
    python corridor/generate_corridor_routes.py --list     # show the scenarios

Output:
    corridor/routes/corridor<N>/<scenario>_seed<S>.rou.xml

CALIBRATION - WHERE THESE NUMBERS COME FROM
-------------------------------------------
The brief asks for demand seeded from real flow profiles.  PeMS, which the
brief names, is *freeway* loop-detector data: no intersections, no turning
movements, no urban arterial geometry.  Calibrating a signalised corridor from
freeway mainline counts would be measuring the wrong quantity on the wrong
facility, so it is not used.  CORRIDOR_PLAN.md section 2 sets this out.

Instead the demand is calibrated from `cologne3`, the RESCO benchmark corridor
bundled with sumo-rl and derived from the TAPAS Cologne mobility model.  It is
a real three-signal corridor with real vehicle-by-vehicle departures.  Measured
from `cologne3.rou.xml` (reproduce with `--provenance`):

    busiest hour (06:31-07:59)      2856 veh/h across the corridor
    dominant entry #1                550 veh/h  (19.3% of demand)
    dominant entry #2                454 veh/h  (15.9%)
    distinct entry edges               33; the top 10 carry 85%

The two dominant entries are the arterial ends; the long tail is side streets.
That gives the shape we need: a heavy arterial fed by many light side streets.
Rounding the arterial to 500 veh/h per end and the side streets to 200 veh/h
per stub yields 2200 veh/h for a three-junction corridor - the same order as
the real 2856, and deliberately a little below it.

WHY CALIBRATION AND NOT LITERAL REPLAY
--------------------------------------
Three reasons, in order of importance:

  1. The whole measurement discipline of this project rests on running several
     stochastic seeds per scenario and reporting the spread.  A literal replay
     is a single sample with no variance estimate.
  2. Calibrated generation is what makes the demand-shift robustness
     experiments possible: `peak` and `reverse` are the base profile moved,
     which a fixed recording cannot give.
  3. Replay onto our synthetic geometry would need the real routes remapped
     onto edges that do not correspond to them, which invents exactly the
     structure it pretends to preserve.

`cologne3` is still evaluated directly, unmodified, as the topology-transfer
target - so the real data is used both ways: as a calibration source here and
as a held-out network in `corridor/robustness.py`.

WHAT IS INHERITED RATHER THAN MEASURED
--------------------------------------
The 60/20/20 straight/left/right split is carried over unchanged from the
single-intersection study.  `cologne3` gives volumes, not turning fractions for
our geometry, so this is an assumption, not a calibration.  Stated here and in
CORRIDOR_REPORT.md rather than left for a reader to discover.
"""

from __future__ import annotations

import argparse
import random
import sys
import zlib
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corridor import build_corridor as bc  # noqa: E402
from corridor import paths  # noqa: E402

DEFAULT_DURATION = 3600
DEFAULT_JUNCTIONS = 3

# Training uses seeds 0-9, evaluation 100-104.  The two sets never overlap, so
# every reported number is measured on demand the agent has never seen - the
# same discipline as the single-intersection study.
TRAINING_SEEDS = list(range(0, 10))
EVALUATION_SEEDS = list(range(100, 105))
TRAINING_SCENARIO = "calibrated"

# Identical to networks/generate_routes.py, deliberately.
TURN_SPLIT = {"straight": 0.60, "left": 0.20, "right": 0.20}

VEHICLE_TYPE = ('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
                'length="5.0"\n           minGap="2.5" maxSpeed="13.89" '
                'carFollowModel="Krauss"/>')

# Flows are given per ENTRY STUB, in vehicles per hour:
#   arterial - each of the two corridor ends (W and E)
#   side     - each of the 2N side-street stubs
SCENARIOS = {
    "calibrated": {
        "arterial": 500, "side": 200,
        "note": "Calibrated from cologne3's busiest hour (see module docstring). "
                "Heavy arterial, light side streets. THIS IS THE TRAINING "
                "SCENARIO.",
    },
    "balanced": {
        "arterial": 350, "side": 350,
        "note": "Equal demand on the arterial and the side streets - the case a "
                "symmetric fixed-time plan is designed for, and so the hardest "
                "for an adaptive controller to beat.",
    },
    "peak": {
        "arterial": 650, "side": 260,
        "note": "ROBUSTNESS: calibrated scaled up 30%. The arterial through "
                "lane runs at roughly its fixed-time capacity, so queues build "
                "faster than they clear.",
    },
    "reverse": {
        "arterial": 200, "side": 500,
        "note": "ROBUSTNESS: the demand shift. Side streets now dominate the "
                "arterial the agent learned to favour - the corridor analogue "
                "of the single-intersection ew_heavy test.",
    },
}


# ---------------------------------------------------------------------------
# Routing over the corridor graph
# ---------------------------------------------------------------------------
def build_graph(n: int) -> dict:
    """Adjacency map node -> [(neighbour, edge id)], both directions."""
    graph = {}
    for a, b in bc.edge_pairs(n):
        graph.setdefault(a, []).append((b, bc.edge_id(a, b)))
        graph.setdefault(b, []).append((a, bc.edge_id(b, a)))
    return graph


def path_edges(n: int, start: str, goal: str) -> list:
    """
    The edge sequence from one boundary node to another.

    Breadth-first rather than a hand-written formula: the corridor is a tree,
    so BFS is exact, and it stays correct if the topology is ever extended.
    A hand-rolled index calculation is the kind of thing that silently returns
    a plausible wrong answer when N changes.
    """
    if start == goal:
        raise ValueError("a trip needs distinct endpoints, got %r twice" % start)
    graph = build_graph(n)
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        node, route = queue.popleft()
        for nxt, edge in graph.get(node, []):
            if nxt in seen:
                continue
            if nxt == goal:
                return route + [edge]
            seen.add(nxt)
            queue.append((nxt, route + [edge]))
    raise ValueError("no path from %s to %s in a %d-junction corridor"
                     % (start, goal, n))


def boundary_nodes(n: int) -> dict:
    """
    Every node traffic can enter or leave by, grouped by role.

    'arterial' is the two corridor ends; 'side' is the 2N street stubs.
    """
    side = []
    for i in range(1, n + 1):
        side.extend(["N%d" % i, "S%d" % i])
    return {"arterial": ["W", "E"], "side": side}


def destinations(n: int, origin: str) -> dict:
    """
    Where a vehicle entering at `origin` may be going, keyed by manoeuvre.

    Mirrors the single-intersection 60/20/20 split at corridor scale:

      arterial entry - "straight" is the full traverse to the far end; "left"
        and "right" exit onto a side street, on the correct side for the
        direction of travel.
      side entry - "straight" crosses the arterial to the opposite stub;
        "left" and "right" turn onto the arterial and run to its end.

    The side choices are lists because a vehicle entering from the west may
    leave at any of the N junctions; one is drawn uniformly at sampling time.
    """
    north = ["N%d" % i for i in range(1, n + 1)]
    south = ["S%d" % i for i in range(1, n + 1)]

    if origin == "W":       # heading east: left is north, right is south
        return {"straight": ["E"], "left": north, "right": south}
    if origin == "E":       # heading west: left is south, right is north
        return {"straight": ["W"], "left": south, "right": north}

    index = int(origin[1:])
    if origin.startswith("N"):   # heading south: left is east, right is west
        return {"straight": ["S%d" % index], "left": ["E"], "right": ["W"]}
    if origin.startswith("S"):   # heading north: left is west, right is east
        return {"straight": ["N%d" % index], "left": ["W"], "right": ["E"]}
    raise ValueError("not a boundary node: %r" % origin)


# ---------------------------------------------------------------------------
# Demand sampling
# ---------------------------------------------------------------------------
def scenario_seed(n: int, scenario: str, seed: int) -> int:
    """
    A reproducible RNG seed for one (corridor size, scenario, seed) triple.

    CRC32 of the joined names, exactly like networks/generate_routes.py.
    Python's built-in hash() is salted per process and would produce different
    traffic on every run - defect D1 in the single-intersection study, and not
    a mistake worth making twice.  The corridor size is mixed in so that the
    2-, 3- and 5-junction corridors do not share an arrival pattern.
    """
    return zlib.crc32(("corridor%d_%s_%d" % (n, scenario, seed)).encode("utf-8"))


def entry_flows(n: int, scenario: str) -> dict:
    """Vehicles per hour entering at each boundary node."""
    spec = SCENARIOS[scenario]
    nodes = boundary_nodes(n)
    flows = {node: spec["arterial"] for node in nodes["arterial"]}
    flows.update({node: spec["side"] for node in nodes["side"]})
    return flows


def sample_vehicles(n: int, scenario: str, duration: int,
                    rng: random.Random) -> list:
    """
    Draw every vehicle's departure time, origin and destination.

    Returns (depart, origin, destination) sorted by time; SUMO requires
    vehicles to be listed in departure order.
    """
    turn_names = list(TURN_SPLIT)
    turn_weights = [TURN_SPLIT[t] for t in turn_names]
    vehicles = []

    for origin, veh_per_hour in sorted(entry_flows(n, scenario).items()):
        if veh_per_hour <= 0:
            continue
        mean_gap = 3600.0 / veh_per_hour
        options = destinations(n, origin)
        t = 0.0
        while True:
            # Exponential inter-arrival gaps == a Poisson arrival process.
            t += rng.expovariate(1.0 / mean_gap)
            if t >= duration:
                break
            turn = rng.choices(turn_names, weights=turn_weights, k=1)[0]
            vehicles.append((t, origin, rng.choice(options[turn])))

    vehicles.sort(key=lambda v: v[0])
    return vehicles


def write_scenario(n: int, scenario: str, seed: int, duration: int) -> Path:
    """Write one route file and return its path."""
    rng = random.Random(scenario_seed(n, scenario, seed))
    vehicles = sample_vehicles(n, scenario, duration, rng)

    path = paths.route_file(n, scenario, seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Distinct origin/destination pairs become named routes; vehicles then
    # reference them. Emitting each path once keeps the file small and makes
    # the route set inspectable.
    pairs = sorted({(o, d) for _, o, d in vehicles})
    route_name = {pair: "r_%s_%s" % pair for pair in pairs}

    with path.open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write("<!--\n")
        fh.write("  Generated by corridor/generate_corridor_routes.py\n")
        fh.write("  Corridor : %d junctions\n" % n)
        fh.write("  Scenario : %s (seed %d)\n" % (scenario, seed))
        fh.write("  Demand   : arterial %d veh/h per end, side %d veh/h per stub\n"
                 % (SCENARIOS[scenario]["arterial"], SCENARIOS[scenario]["side"]))
        fh.write("  Vehicles : %d in %d s\n" % (len(vehicles), duration))
        fh.write("  Note     : %s\n" % SCENARIOS[scenario]["note"])
        fh.write("-->\n")
        fh.write("<routes>\n")
        fh.write(VEHICLE_TYPE + "\n")
        for pair in pairs:
            fh.write('    <route id="%s" edges="%s"/>\n'
                     % (route_name[pair], " ".join(path_edges(n, *pair))))
        for i, (depart, origin, dest) in enumerate(vehicles):
            fh.write('    <vehicle id="v%d" type="car" route="%s" depart="%.2f"'
                     ' departLane="best"/>\n'
                     % (i, route_name[(origin, dest)], depart))
        fh.write("</routes>\n")
    return path


def all_seeds() -> list:
    return TRAINING_SEEDS + EVALUATION_SEEDS


def training_route_files(n: int = DEFAULT_JUNCTIONS) -> list:
    return [paths.route_file(n, TRAINING_SCENARIO, s) for s in TRAINING_SEEDS]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def provenance() -> None:
    """Recompute the cologne3 calibration figures quoted in the docstring."""
    import collections
    import xml.etree.ElementTree as ET

    rou = (paths.ROOT / ".venv" / "Lib" / "site-packages" / "sumo_rl" / "nets"
           / "RESCO" / "cologne3" / "cologne3.rou.xml")
    if not rou.exists():
        print("cologne3 not found at %s" % rou)
        return
    root = ET.parse(rou).getroot()
    busiest = [v for v in root.iter("vehicle")
               if 25200 <= float(v.get("depart")) < 28800]
    first = collections.Counter(
        v.find("route").get("edges").split()[0] for v in busiest)
    total = sum(first.values())
    print("cologne3 busiest hour: %d veh/h across the corridor" % total)
    print("distinct entry edges : %d" % len(first))
    for edge, count in first.most_common(3):
        print("  %-28s %4d veh/h  (%.1f%%)" % (edge, count, 100 * count / total))
    top10 = sum(c for _, c in first.most_common(10))
    print("top-10 entries carry %.0f%% of demand" % (100 * top10 / total))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate corridor demand.")
    parser.add_argument("--junctions", type=int, default=DEFAULT_JUNCTIONS)
    parser.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--list", action="store_true", help="show the scenarios")
    parser.add_argument("--provenance", action="store_true",
                        help="recompute the cologne3 calibration figures")
    args = parser.parse_args()

    if args.provenance:
        provenance()
        return

    if args.list:
        n = args.junctions
        print("Corridor demand scenarios (%d junctions)\n" % n)
        for name, spec in SCENARIOS.items():
            total = sum(entry_flows(n, name).values())
            print("  %-12s arterial %4d  side %4d  total %5d veh/h"
                  % (name, spec["arterial"], spec["side"], total))
            print("               %s\n" % spec["note"])
        return

    seeds = args.seeds if args.seeds is not None else all_seeds()
    written = 0
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise SystemExit("unknown scenario %r; choose from %s"
                             % (scenario, ", ".join(sorted(SCENARIOS))))
        for seed in seeds:
            write_scenario(args.junctions, scenario, seed, args.duration)
            written += 1
    print("Wrote %d route files to %s"
          % (written, (paths.ROUTES_DIR / paths.corridor_name(args.junctions))
             .relative_to(paths.ROOT).as_posix()))


if __name__ == "__main__":
    main()
