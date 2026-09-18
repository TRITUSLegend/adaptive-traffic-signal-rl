"""
Build an N-intersection arterial corridor and install signal programs on it.

Run:
    python corridor/build_corridor.py                 # the 3-junction corridor
    python corridor/build_corridor.py --junctions 2   # transfer target
    python corridor/build_corridor.py --junctions 5   # transfer target

Output, per corridor size:
    corridor/networks/corridor<N>/corridor<N>.net.xml           static plans
    corridor/networks/corridor<N>/corridor<N>_actuated.net.xml  actuated plans

WHY THIS SCRIPT EXISTS
----------------------
It is `networks/build_network.py` generalised from one junction to N, and it
deliberately reuses that script's two good decisions:

  1. Write the node/edge/connection description in plain XML, compile it with
     netconvert, then *replace* the auto-generated signal program with an
     explicit one.  netconvert's own program differs between SUMO versions and
     lumps left turns in with through traffic; neither is acceptable in a
     controlled experiment.
  2. Never hard-code link indices.  netconvert assigns them and the ordering is
     not ours to assume, so the compiled network is read back and each phase
     string is built from the connections' own `linkIndex` and `dir`.

GEOMETRY
--------
A linear west-east arterial with N signalised junctions, each also served by a
north and a south side street:

            N1          N2          N3
            |           |           |
      W --[J1]--400m--[J2]--400m--[J3]-- E
            |           |           |
            S1          S2          S3

Spacing is 400 m between adjacent signals: long enough to hold a queue without
spilling back into the upstream junction, which would suppress demand and
flatter every controller equally.  Outer arms are 250 m, the same length the
single-intersection study chose (~33 vehicles per lane of storage, far more
than we ever queue).

EDGE NAMING
-----------
`<from>_<to>` using node ids, e.g. `W_J1`, `J1_J2`, `N1_J1`.

This departs from the single-intersection convention (`N2C` for north-into-
centre).  That convention does not survive generalisation: `J12J2` cannot be
read unambiguously as J1->J2 rather than J12->J2.  An explicit separator is
worth the inconsistency, and nothing imports names across the two studies.

THE FOUR PHASES, PER JUNCTION
-----------------------------
Identical in structure to the single-intersection plan, so a policy trained on
one junction has the same action space as a policy trained here:

    0  SIDE-through   north-south straight + right
    1  SIDE-left      north-south protected left
    2  ART-through    east-west (arterial) straight + right
    3  ART-left       east-west protected left

Phases are mutually exclusive, so every phase is collision-free and there are
no permissive give-way lefts.  That keeps gap-acceptance behaviour out of the
comparison: any difference we measure comes from signal timing alone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sumo_setup  # noqa: E402  (must run before any SUMO import)

from corridor import paths  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
SPACING_M = 400          # between adjacent signalised junctions
ARM_M = 250              # outer arterial stubs and every side street
LANES = 2                # lane 0 = straight+right, lane 1 = protected left
SPEED_MPS = 13.89        # 50 km/h, matching the single-intersection network

# Signal timings for the FIXED-TIME baseline, in seconds.
#
# Identical to networks/build_network.py so that the corridor's fixed-time plan
# is the same controller the single-intersection study measured, just repeated.
# The cycle is 104 s: (32+4) + (12+4) + (32+4) + (12+4).
GREEN_THROUGH = 32
GREEN_LEFT = 12
YELLOW = 4

# Bounds for the ACTUATED baseline.
#
# These are pinned to the same values the rest of the project uses so that
# every controller pays an identical safety cost - the project's rule that
# differing amber or minimum green would be the easiest way to rig a
# comparison.  MAX_DUR matches SafetyConstrainedController.MAX_GREEN.
MIN_DUR = 10             # = MIN_GREEN
MAX_DUR = 60             # = SafetyConstrainedController.MAX_GREEN
MAX_GAP = 3.0            # seconds of headway that terminates a green
DETECTOR_GAP = 2.0

PHASE_NAMES = ["SIDE-through", "SIDE-left", "ART-through", "ART-left"]
TURN_LABEL = {"s": "straight", "l": "left", "r": "right", "t": "u-turn"}


# ---------------------------------------------------------------------------
# Topology description
# ---------------------------------------------------------------------------
def junction_ids(n: int) -> list:
    return ["J%d" % (i + 1) for i in range(n)]


def node_layout(n: int) -> list:
    """
    Every node as (id, x, y, is_signal).

    The arterial runs along y=0 from west to east.  Side-street stubs sit
    ARM_M north and south of each junction.
    """
    nodes = [("W", -ARM_M, 0, False)]
    for i, jid in enumerate(junction_ids(n)):
        x = i * SPACING_M
        nodes.append((jid, x, 0, True))
        nodes.append(("N%d" % (i + 1), x, ARM_M, False))
        nodes.append(("S%d" % (i + 1), x, -ARM_M, False))
    nodes.append(("E", (n - 1) * SPACING_M + ARM_M, 0, False))
    return nodes


def edge_pairs(n: int) -> list:
    """
    Every bidirectional road as an (a, b) node pair; both directions are built.

    The arterial is the chain W - J1 - ... - Jn - E; each junction also has a
    north and a south stub.
    """
    chain = ["W"] + junction_ids(n) + ["E"]
    pairs = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    for i, jid in enumerate(junction_ids(n)):
        pairs.append(("N%d" % (i + 1), jid))
        pairs.append(("S%d" % (i + 1), jid))
    return pairs


def edge_id(a: str, b: str) -> str:
    return "%s_%s" % (a, b)


def approaches(n: int, junction: str) -> dict:
    """
    The four inbound edges of one junction, keyed by compass direction.

    'west' is the edge arriving from the west - i.e. traffic heading east.
    """
    order = ["W"] + junction_ids(n) + ["E"]
    i = order.index(junction)
    k = junction_ids(n).index(junction) + 1
    return {
        "west": edge_id(order[i - 1], junction),
        "east": edge_id(order[i + 1], junction),
        "north": edge_id("N%d" % k, junction),
        "south": edge_id("S%d" % k, junction),
    }


def exits(n: int, junction: str) -> dict:
    """The four outbound edges of one junction, keyed by compass direction."""
    order = ["W"] + junction_ids(n) + ["E"]
    i = order.index(junction)
    k = junction_ids(n).index(junction) + 1
    return {
        "west": edge_id(junction, order[i - 1]),
        "east": edge_id(junction, order[i + 1]),
        "north": edge_id(junction, "N%d" % k),
        "south": edge_id(junction, "S%d" % k),
    }


# Where each turn takes you, by the direction you arrived FROM.
#
# Read it as: a vehicle arriving from the west is heading east, so its
# "straight" exit is east, its "left" is north and its "right" is south.
#
# This table mirrors networks/generate_routes.py:TURNS, which is the
# known-correct reference:
#
#     "N2C": right -> C2W,  straight -> C2S,  left -> C2E
#     "S2C": right -> C2E,  straight -> C2N,  left -> C2W
#     "E2C": right -> C2N,  straight -> C2W,  left -> C2S
#     "W2C": right -> C2S,  straight -> C2E,  left -> C2N
#
# The first version of this table had north and south swapped, which is an
# easy mistake to make from a mental picture and a hard one to see afterwards:
# the network still compiled, traffic still flowed, and the metrics were
# plausible - but netconvert had assigned `dir=l` to a movement this file had
# placed in the through-and-right lane, so the protected left-turn lane was
# carrying right turns and the phase groups released the wrong movements.
# `tests/test_corridor_network.py` now checks every junction's compiled `dir`
# against this table, so the same slip fails loudly next time.
TURN_TARGETS = {
    "west":  {"straight": "east",  "left": "north", "right": "south"},
    "east":  {"straight": "west",  "left": "south", "right": "north"},
    "north": {"straight": "south", "left": "east",  "right": "west"},
    "south": {"straight": "north", "left": "west",  "right": "east"},
}

# Which approaches hold green together.  The arterial pair and the side-street
# pair are the two non-conflicting groups, exactly as NS/EW are at a single
# junction.
SIDE_DIRECTIONS = ("north", "south")
ARTERIAL_DIRECTIONS = ("west", "east")


# ---------------------------------------------------------------------------
# XML emission
# ---------------------------------------------------------------------------
def write_nodes(path: Path, n: int) -> None:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             "<!-- generated by corridor/build_corridor.py - do not hand-edit -->",
             "<nodes>"]
    for nid, x, y, is_signal in node_layout(n):
        if is_signal:
            lines.append(
                '    <node id="%s" x="%d" y="%d" type="traffic_light" '
                'tlType="static" tl="%s"/>' % (nid, x, y, nid))
        else:
            lines.append('    <node id="%s" x="%d" y="%d" type="priority"/>'
                         % (nid, x, y))
    lines.append("</nodes>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_edges(path: Path, n: int) -> None:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             "<!-- generated by corridor/build_corridor.py - do not hand-edit -->",
             "<edges>"]
    for a, b in edge_pairs(n):
        for frm, to in ((a, b), (b, a)):
            lines.append(
                '    <edge id="%s" from="%s" to="%s" numLanes="%d" '
                'speed="%.2f" priority="1"/>'
                % (edge_id(frm, to), frm, to, LANES, SPEED_MPS))
    lines.append("</edges>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_connections(path: Path, n: int) -> None:
    """
    Pin the lane function at every junction: lane 0 straight+right, lane 1 left.

    Writing these by hand rather than letting netconvert infer them is what
    makes the corridor's movement structure identical to the single
    intersection's, which is in turn what makes a policy transferable between
    them.
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             "<!-- generated by corridor/build_corridor.py - do not hand-edit -->",
             "<connections>"]
    for jid in junction_ids(n):
        inbound = approaches(n, jid)
        outbound = exits(n, jid)
        for direction, in_edge in inbound.items():
            targets = TURN_TARGETS[direction]
            # lane 0 carries straight ahead and the right turn
            lines.append('    <connection from="%s" to="%s" fromLane="0" toLane="0"/>'
                         % (in_edge, outbound[targets["straight"]]))
            lines.append('    <connection from="%s" to="%s" fromLane="0" toLane="1"/>'
                         % (in_edge, outbound[targets["right"]]))
            # lane 1 is the protected left
            lines.append('    <connection from="%s" to="%s" fromLane="1" toLane="1"/>'
                         % (in_edge, outbound[targets["left"]]))
    lines.append("</connections>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# netconvert
# ---------------------------------------------------------------------------
def run_netconvert(work: Path, out: Path) -> None:
    cmd = [
        sumo_setup.binary("netconvert"),
        "--node-files", str(work / "nodes.nod.xml"),
        "--edge-files", str(work / "edges.edg.xml"),
        "--connection-files", str(work / "connections.con.xml"),
        "--output-file", str(out),
        "--no-turnarounds", "true",
        "--offset.disable-normalization", "true",
        "--tls.default-type", "static",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("netconvert failed with exit code %d" % result.returncode)


def read_controlled_links(net_path: Path) -> dict:
    """
    Map traffic-light id -> {link index: (incoming edge, turn direction)}.

    The single-intersection version returned one flat dict because there was
    one signal.  Here the same information has to be kept per junction.
    """
    tree = ET.parse(net_path)
    per_tl = {}
    for conn in tree.getroot().iter("connection"):
        tl = conn.get("tl")
        if tl is None:
            continue
        per_tl.setdefault(tl, {})[int(conn.get("linkIndex"))] = (
            conn.get("from"), conn.get("dir"))
    if not per_tl:
        raise SystemExit("No traffic-light-controlled connections found.")
    return per_tl


def build_phase_states(links: dict, inbound: dict) -> list:
    """
    The four green strings for one junction.

    `links` maps link index -> (incoming edge, turn dir) for this junction.
    `inbound` maps compass direction -> incoming edge id.
    """
    n_links = max(links) + 1
    side_edges = {inbound[d] for d in SIDE_DIRECTIONS}
    arterial_edges = {inbound[d] for d in ARTERIAL_DIRECTIONS}

    def make(edges: set, directions: set) -> str:
        chars = []
        for i in range(n_links):
            edge, direction = links.get(i, ("", ""))
            chars.append("G" if edge in edges and direction in directions else "r")
        return "".join(chars)

    return [
        make(side_edges, {"s", "r"}),        # 0: side-street through + right
        make(side_edges, {"l"}),             # 1: side-street left
        make(arterial_edges, {"s", "r"}),    # 2: arterial through + right
        make(arterial_edges, {"l"}),         # 3: arterial left
    ]


def yellow_state(green: str) -> str:
    """Every movement that had green turns amber; everything else stays red."""
    return "".join("y" if c == "G" else "r" for c in green)


def phase_durations() -> list:
    return [GREEN_THROUGH, GREEN_LEFT, GREEN_THROUGH, GREEN_LEFT]


def write_tl_programs(net_path: Path, n: int, actuated: bool) -> dict:
    """
    Replace every auto-generated signal program with an explicit one.

    Returns {junction id: [green strings]} so callers can report or test it.

    The actuated variant keeps the *same* phase structure and order and changes
    only how long each phase runs: instead of a fixed duration it gets a
    min/max duration and SUMO's gap-out logic.  That isolation is deliberate -
    fixed-time and actuated differ in their timing rule and in nothing else, so
    the comparison between them measures responsiveness rather than a different
    phase design.
    """
    tree = ET.parse(net_path)
    root = tree.getroot()
    per_tl = read_controlled_links(net_path)
    greens_by_junction = {}

    for logic in list(root.iter("tlLogic")):
        tl_id = logic.get("id")
        inbound = approaches(n, tl_id)
        greens = build_phase_states(per_tl[tl_id], inbound)
        greens_by_junction[tl_id] = greens

        for child in list(logic):
            logic.remove(child)
        logic.set("type", "actuated" if actuated else "static")
        logic.set("programID", "actuated" if actuated else "fixed_time")
        logic.set("offset", "0")

        if actuated:
            # SUMO builds the lane-area detectors it needs for gap-out itself.
            for key, value in (("max-gap", MAX_GAP),
                               ("detector-gap", DETECTOR_GAP),
                               ("show-detectors", "false")):
                param = ET.SubElement(logic, "param")
                param.set("key", key)
                param.set("value", str(value))

        for name, green, duration in zip(PHASE_NAMES, greens, phase_durations()):
            phase = ET.SubElement(logic, "phase")
            phase.set("duration", str(duration))
            phase.set("state", green)
            if actuated:
                phase.set("minDur", str(MIN_DUR))
                phase.set("maxDur", str(MAX_DUR))
            phase.set("name", name)
            amber = ET.SubElement(logic, "phase")
            amber.set("duration", str(YELLOW))
            amber.set("state", yellow_state(green))
            amber.set("name", name + "-yellow")

    ET.indent(tree, space="    ")
    tree.write(net_path, encoding="UTF-8", xml_declaration=True)
    return greens_by_junction


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build(n: int, quiet: bool = False) -> dict:
    """Build both network variants for an N-junction corridor."""
    out_dir = paths.network_dir(n)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_nodes(out_dir / "nodes.nod.xml", n)
    write_edges(out_dir / "edges.edg.xml", n)
    write_connections(out_dir / "connections.con.xml", n)

    built = {}
    for actuated in (False, True):
        net = paths.net_file(n, actuated=actuated)
        run_netconvert(out_dir, net)
        built[("actuated" if actuated else "static")] = write_tl_programs(
            net, n, actuated=actuated)
        if not quiet:
            print("  wrote %s" % net.relative_to(paths.ROOT).as_posix())
    return built


def print_summary(n: int, greens_by_junction: dict) -> None:
    print("\nCorridor with %d signalised junctions" % n)
    print("  spacing %d m, outer arms %d m, %d lanes per approach"
          % (SPACING_M, ARM_M, LANES))
    for jid in junction_ids(n):
        greens = greens_by_junction[jid]
        print("\n  %s" % jid)
        for name, green, dur in zip(PHASE_NAMES, greens, phase_durations()):
            print("    %-13s %3ds  %s" % (name, dur, green))
    print("\n  cycle length = %ds" % (sum(phase_durations()) + 4 * YELLOW))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an N-intersection arterial corridor network.")
    parser.add_argument("--junctions", type=int, default=3,
                        help="number of signalised junctions (default 3)")
    parser.add_argument("--all", action="store_true",
                        help="build the 2, 3 and 5 junction corridors")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(sumo_setup.describe())
    sizes = [2, 3, 5] if args.all else [args.junctions]
    for n in sizes:
        built = build(n, quiet=args.quiet)
        if not args.quiet:
            print_summary(n, built["static"])


if __name__ == "__main__":
    main()
