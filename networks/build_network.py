"""
Build the single-intersection SUMO network, then install a hand-written
traffic-light program on it.

Run:
    python networks/build_network.py

Output:
    networks/single_intersection.net.xml

WHY THIS SCRIPT EXISTS
----------------------
``netconvert`` (SUMO's network compiler) turns our plain-text node/edge/
connection descriptions into a full network file.  It will also invent a
traffic-light program for us, but that auto-generated program differs between
SUMO versions and lumps left turns in with through traffic.  Both are bad for
a controlled experiment, so after netconvert runs we *replace* the signal
program with an explicit 4-phase plan defined here.

THE FOUR SIGNAL PHASES
----------------------
A "phase" is one combination of green and red lights.  Choosing which of these
four to show at any moment is the agent's entire job:

    0  NS-through   north-south straight-ahead and right turns
    1  NS-left      north-south protected left turns
    2  EW-through   east-west straight-ahead and right turns
    3  EW-left      east-west protected left turns

The movements inside a phase never cross each other's path, so every phase is
collision-free.  Because the phases are mutually exclusive the intersection is
"fully protected": there are no permissive give-way left turns.  That keeps
gap-acceptance behaviour out of the comparison, so any performance difference
we measure comes from signal timing alone.

WHY A LINK-INDEX LOOKUP IS NEEDED
---------------------------------
SUMO encodes a phase as a string of characters, one per controlled movement,
e.g. "GGrrrrGGrrrr".  Position i in that string is the light shown to "link
index" i.  netconvert assigns those indices itself and the ordering is not
something we should hard-code.  So we read the generated network back, ask
each connection for its link index and turn direction, and build the phase
strings programmatically.  This is version-proof.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Make the repo root importable so `import sumo_setup` works regardless of the
# directory this script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sumo_setup  # noqa: E402  (must run before any SUMO import)

HERE = Path(__file__).resolve().parent
NET_FILE = HERE / "single_intersection.net.xml"

# The four approaches, grouped into the two opposing pairs that can safely hold
# green at the same time.
NORTH_SOUTH = {"N2C", "S2C"}
EAST_WEST = {"E2C", "W2C"}

# Signal timings for the FIXED-TIME baseline controller, in seconds.
#
# These follow ordinary traffic-engineering practice for a 50 km/h urban
# intersection.  Through movements get much more green than left turns because
# they carry most of the traffic.  Yellow is 4 s, the usual value at this speed
# limit (roughly the time needed to stop safely from 50 km/h).
#
# Total cycle = (32+4) + (12+4) + (32+4) + (12+4) = 104 s, inside the normal
# 90-120 s band for a signalised urban intersection.
#
# NOTE: this plan is deliberately SYMMETRIC - north-south and east-west get
# identical green time.  That is what "an untuned fixed-time signal" means and
# it is the baseline the project brief asks for.  Its weakness under lopsided
# demand is exactly the gap the RL agent should close, so `baseline/` also
# provides a demand-tuned fixed plan to keep the comparison honest.
GREEN_THROUGH = 32
GREEN_LEFT = 12
YELLOW = 4

PHASE_NAMES = ["NS-through", "NS-left", "EW-through", "EW-left"]

TURN_LABEL = {"s": "straight", "l": "left", "r": "right", "t": "u-turn"}


def run_netconvert() -> None:
    """Compile the node / edge / connection files into a .net.xml network."""
    cmd = [
        sumo_setup.binary("netconvert"),
        "--node-files", str(HERE / "nodes.nod.xml"),
        "--edge-files", str(HERE / "edges.edg.xml"),
        "--connection-files", str(HERE / "connections.con.xml"),
        "--output-file", str(NET_FILE),
        # Do not invent U-turns at the junction: they are not part of our phase
        # design and would create movements that no phase ever serves.
        "--no-turnarounds", "true",
        # Deterministic output, so re-running produces an identical network.
        "--offset.disable-normalization", "true",
        "--tls.default-type", "static",
    ]
    print("$ netconvert (building single_intersection.net.xml)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"netconvert failed with exit code {result.returncode}")
    for line in result.stderr.splitlines():
        if line.strip():
            print("  netconvert:", line)


def read_controlled_links(net_path: Path) -> dict:
    """
    Map each traffic-light link index to (incoming edge, turn direction).

    Turn direction is SUMO's own single-letter code:
    's' straight, 'l' left, 'r' right, 't' turnaround.
    """
    tree = ET.parse(net_path)
    links = {}
    for conn in tree.getroot().iter("connection"):
        if conn.get("tl") is None:
            continue  # this movement is not signal-controlled
        links[int(conn.get("linkIndex"))] = (conn.get("from"), conn.get("dir"))
    if not links:
        raise SystemExit("No traffic-light-controlled connections found in the network.")
    return links


def build_phase_states(links: dict) -> list:
    """
    Turn the link table into the four green-phase strings.

    A phase string has one character per link index:
        'G' = protected green (has right of way)
        'r' = red
    """
    n_links = max(links) + 1

    def make(approaches: set, directions: set) -> str:
        chars = []
        for i in range(n_links):
            edge, direction = links.get(i, ("", ""))
            has_green = edge in approaches and direction in directions
            chars.append("G" if has_green else "r")
        return "".join(chars)

    return [
        make(NORTH_SOUTH, {"s", "r"}),  # 0: NS-through (plus right turns)
        make(NORTH_SOUTH, {"l"}),       # 1: NS-left
        make(EAST_WEST, {"s", "r"}),    # 2: EW-through (plus right turns)
        make(EAST_WEST, {"l"}),         # 3: EW-left
    ]


def yellow_state(green: str) -> str:
    """
    The yellow (amber) phase that follows a given green phase.

    Every movement that had green turns amber; everything else stays red.
    That is safe here because our four phases are mutually exclusive, so a
    movement that is green now is red in whichever phase comes next.
    """
    return "".join("y" if c == "G" else "r" for c in green)


def write_tl_program(net_path: Path, greens: list) -> None:
    """Replace the auto-generated signal program with our explicit 4-phase plan."""
    tree = ET.parse(net_path)
    logic = tree.getroot().find("tlLogic")
    if logic is None:
        raise SystemExit("Network has no <tlLogic>; is node C typed traffic_light?")

    # Drop netconvert's phases and write ours: green, yellow, green, yellow, ...
    for child in list(logic):
        logic.remove(child)
    logic.set("type", "static")
    logic.set("programID", "fixed_time")
    logic.set("offset", "0")

    for name, green, duration in zip(PHASE_NAMES, greens, phase_durations()):
        phase = ET.SubElement(logic, "phase")
        phase.set("duration", str(duration))
        phase.set("state", green)
        phase.set("name", name)
        amber = ET.SubElement(logic, "phase")
        amber.set("duration", str(YELLOW))
        amber.set("state", yellow_state(green))
        amber.set("name", name + "-yellow")

    ET.indent(tree, space="    ")
    tree.write(net_path, encoding="UTF-8", xml_declaration=True)


def phase_durations() -> list:
    """Green durations of the four phases, in the order they are served."""
    return [GREEN_THROUGH, GREEN_LEFT, GREEN_THROUGH, GREEN_LEFT]


def print_summary(links: dict, greens: list) -> None:
    print("\nNetwork written to " + str(NET_FILE))
    print(str(len(links)) + " signal-controlled movements at junction C\n")
    print("  idx  approach  turn")
    for i in sorted(links):
        edge, direction = links[i]
        print("  %3d  %-8s  %s" % (i, edge, TURN_LABEL.get(direction, direction)))
    print("\nFixed-time program (the baseline controller):")
    for name, green, dur in zip(PHASE_NAMES, greens, phase_durations()):
        print("  %-11s %3ds  %s" % (name, dur, green))
        print("  %-11s %3ds  %s" % ("  + yellow", YELLOW, yellow_state(green)))
    print("\n  cycle length = %ds" % (sum(phase_durations()) + 4 * YELLOW))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the 4-way intersection network and its signal program."
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the phase table")
    args = parser.parse_args()

    print(sumo_setup.describe())
    run_netconvert()

    links = read_controlled_links(NET_FILE)
    greens = build_phase_states(links)
    write_tl_program(NET_FILE, greens)

    if not args.quiet:
        print_summary(links, greens)


if __name__ == "__main__":
    main()
