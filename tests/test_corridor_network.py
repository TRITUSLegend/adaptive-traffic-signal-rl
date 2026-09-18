"""
Tests for the corridor network built by corridor/build_corridor.py.

WHY THESE TESTS EXIST
---------------------
The same reason as tests/test_network_phases.py, multiplied by the number of
junctions: the phase strings ARE the safety design, netconvert decides which
movement each character refers to, and a wrong mapping produces a junction that
could not exist in reality while the simulation runs happily and reports
plausible numbers.

One of these tests exists because that failure actually happened here. The
first version of `TURN_TARGETS` had north and south swapped. The corridor
compiled, traffic flowed, and the delay figures looked reasonable - but
netconvert had labelled `dir=l` a movement the generator had placed in the
through-and-right lane, so the protected left-turn lane carried right turns and
each phase released the wrong set of movements. Nothing failed; the geometry
was simply wrong. `test_every_left_turn_is_on_the_left_lane` and
`test_turn_targets_match_the_single_intersection_reference` are the two checks
that would have caught it at the moment it was written.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor import paths  # noqa: E402
from corridor.build_corridor import (  # noqa: E402
    ARM_M,
    ARTERIAL_DIRECTIONS,
    GREEN_LEFT,
    GREEN_THROUGH,
    MAX_DUR,
    MIN_DUR,
    PHASE_NAMES,
    SIDE_DIRECTIONS,
    SPACING_M,
    TURN_TARGETS,
    YELLOW,
    approaches,
    build_phase_states,
    edge_id,
    edge_pairs,
    exits,
    junction_ids,
    node_layout,
    phase_durations,
    read_controlled_links,
    yellow_state,
)

N = 3

# The single-intersection network's phase plan, which the corridor's junctions
# must reproduce exactly. This is not a coincidence to be tolerated - it is the
# property that makes a policy transferable between the two studies, because it
# means a corridor junction presents the same action space and the same
# movement structure as the intersection the original agents trained on.
SINGLE_INTERSECTION_PHASES = [
    "GGrrrrGGrrrr",
    "rrGrrrrrGrrr",
    "rrrGGrrrrGGr",
    "rrrrrGrrrrrG",
]


@pytest.fixture(scope="module")
def net_path():
    path = paths.net_file(N)
    if not path.exists():
        pytest.skip("corridor not built - run: python corridor/build_corridor.py")
    return path


@pytest.fixture(scope="module")
def links(net_path):
    return read_controlled_links(net_path)


@pytest.fixture(scope="module")
def connections(net_path):
    """Every signal-controlled connection as a list of dicts."""
    rows = []
    for conn in ET.parse(net_path).getroot().iter("connection"):
        if conn.get("tl") is None:
            continue
        rows.append({
            "tl": conn.get("tl"),
            "index": int(conn.get("linkIndex")),
            "from": conn.get("from"),
            "to": conn.get("to"),
            "from_lane": int(conn.get("fromLane")),
            "dir": conn.get("dir"),
        })
    return rows


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
def test_the_corridor_has_the_requested_number_of_signals(links):
    assert sorted(links) == junction_ids(N)


def test_junctions_are_evenly_spaced_along_the_arterial():
    xs = [x for nid, x, y, sig in node_layout(N) if sig]
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    assert gaps == [SPACING_M] * (N - 1)


def test_signalised_junctions_sit_on_the_arterial_centreline():
    assert all(y == 0 for _, _, y, sig in node_layout(N) if sig)


def test_side_streets_are_one_arm_length_from_their_junction():
    layout = {nid: (x, y) for nid, x, y, _ in node_layout(N)}
    for i, jid in enumerate(junction_ids(N), start=1):
        jx, _ = layout[jid]
        assert layout["N%d" % i] == (jx, ARM_M)
        assert layout["S%d" % i] == (jx, -ARM_M)


def test_every_road_is_built_in_both_directions():
    ids = set()
    for a, b in edge_pairs(N):
        ids.add(edge_id(a, b))
        ids.add(edge_id(b, a))
    for a, b in edge_pairs(N):
        assert edge_id(a, b) in ids and edge_id(b, a) in ids


def test_each_junction_has_exactly_four_approaches_and_four_exits():
    for jid in junction_ids(N):
        assert len(set(approaches(N, jid).values())) == 4
        assert len(set(exits(N, jid).values())) == 4


def test_interior_junctions_are_reached_from_their_neighbours():
    """J2's east and west approaches must come from J3 and J1, not from stubs."""
    inbound = approaches(N, "J2")
    assert inbound["west"] == edge_id("J1", "J2")
    assert inbound["east"] == edge_id("J3", "J2")


# ---------------------------------------------------------------------------
# Turn geometry - the regression the swapped-compass bug left behind
# ---------------------------------------------------------------------------
def test_turn_targets_match_the_single_intersection_reference():
    """
    The reference is networks/generate_routes.py:TURNS, which is known correct:

        from north (heading south): left=east,  right=west
        from south (heading north): left=west,  right=east
        from west  (heading east) : left=north, right=south
        from east  (heading west) : left=south, right=north
    """
    assert TURN_TARGETS["north"] == {"straight": "south", "left": "east",
                                     "right": "west"}
    assert TURN_TARGETS["south"] == {"straight": "north", "left": "west",
                                     "right": "east"}
    assert TURN_TARGETS["west"] == {"straight": "east", "left": "north",
                                    "right": "south"}
    assert TURN_TARGETS["east"] == {"straight": "west", "left": "south",
                                    "right": "north"}


def test_a_turn_and_its_opposite_are_never_the_same_exit():
    for direction, targets in TURN_TARGETS.items():
        assert len(set(targets.values())) == 3, direction


def test_every_left_turn_is_on_the_left_lane(connections):
    """
    Lane 1 is the protected left; lane 0 carries straight ahead and the right
    turn. This is the check that fails if the compass mapping is wrong, because
    netconvert derives `dir` from the geometry rather than from our intent.
    """
    for row in connections:
        is_left = row["dir"] == "l"
        on_left_lane = row["from_lane"] == 1
        assert is_left == on_left_lane, (
            "%s link %d (%s -> %s) has dir=%s on lane %d"
            % (row["tl"], row["index"], row["from"], row["to"], row["dir"],
               row["from_lane"]))


def test_each_approach_contributes_exactly_one_left_one_straight_one_right(
        connections):
    for jid in junction_ids(N):
        for approach in set(approaches(N, jid).values()):
            dirs = sorted(r["dir"] for r in connections
                          if r["tl"] == jid and r["from"] == approach)
            assert dirs == ["l", "r", "s"], (jid, approach, dirs)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def test_every_junction_controls_twelve_movements(links):
    for jid, mapping in links.items():
        assert len(mapping) == 12, jid


def test_phase_strings_reproduce_the_single_intersection_plan(links):
    for jid in junction_ids(N):
        greens = build_phase_states(links[jid], approaches(N, jid))
        assert greens == SINGLE_INTERSECTION_PHASES, jid


def test_there_are_four_named_phases():
    assert len(PHASE_NAMES) == 4
    assert len(phase_durations()) == 4


def test_no_two_phases_give_green_to_the_same_movement(links):
    """Mutual exclusivity is what makes every phase collision-free."""
    for jid in junction_ids(N):
        greens = build_phase_states(links[jid], approaches(N, jid))
        for index in range(len(greens[0])):
            lit = [g for g in greens if g[index] == "G"]
            assert len(lit) <= 1, (jid, index)


def test_every_movement_is_served_by_some_phase(links):
    """A movement no phase ever releases would be starved for the whole hour."""
    for jid in junction_ids(N):
        greens = build_phase_states(links[jid], approaches(N, jid))
        for index in range(len(greens[0])):
            assert any(g[index] == "G" for g in greens), (jid, index)


def test_side_and_arterial_phases_release_disjoint_approaches(links):
    for jid in junction_ids(N):
        inbound = approaches(N, jid)
        side = {inbound[d] for d in SIDE_DIRECTIONS}
        arterial = {inbound[d] for d in ARTERIAL_DIRECTIONS}
        assert side.isdisjoint(arterial)
        assert len(side) == 2 and len(arterial) == 2


def test_amber_follows_exactly_the_movements_that_had_green():
    green = "GGrrrrGGrrrr"
    assert yellow_state(green) == "yyrrrryyrrrr"


def test_amber_never_invents_a_green():
    for phase in SINGLE_INTERSECTION_PHASES:
        amber = yellow_state(phase)
        assert "G" not in amber
        assert len(amber) == len(phase)


def test_cycle_length_matches_the_single_intersection_plan():
    assert sum(phase_durations()) + 4 * YELLOW == 104
    assert phase_durations() == [GREEN_THROUGH, GREEN_LEFT,
                                 GREEN_THROUGH, GREEN_LEFT]


# ---------------------------------------------------------------------------
# The compiled programs
# ---------------------------------------------------------------------------
def test_static_network_carries_static_programs(net_path):
    for logic in ET.parse(net_path).getroot().iter("tlLogic"):
        assert logic.get("type") == "static"


def test_actuated_network_carries_actuated_programs():
    path = paths.net_file(N, actuated=True)
    if not path.exists():
        pytest.skip("actuated corridor not built")
    logics = list(ET.parse(path).getroot().iter("tlLogic"))
    assert len(logics) == N
    for logic in logics:
        assert logic.get("type") == "actuated"


def test_actuated_bounds_match_the_projects_safety_constants():
    """
    Every controller must pay an identical safety cost. The actuated plan's
    minimum and maximum green are therefore pinned to the same MIN_GREEN and
    MAX_GREEN the safety wrapper enforces on the learned policies, rather than
    to SUMO's defaults.
    """
    from eval.simulation import MIN_GREEN, SafetyConstrainedController
    assert MIN_DUR == MIN_GREEN
    assert MAX_DUR == SafetyConstrainedController.MAX_GREEN


def test_actuated_green_phases_carry_min_and_max_durations():
    path = paths.net_file(N, actuated=True)
    if not path.exists():
        pytest.skip("actuated corridor not built")
    for logic in ET.parse(path).getroot().iter("tlLogic"):
        greens = [p for p in logic.iter("phase") if "y" not in p.get("state")]
        assert len(greens) == 4
        for phase in greens:
            assert int(phase.get("minDur")) == MIN_DUR
            assert int(phase.get("maxDur")) == MAX_DUR


def test_both_network_variants_describe_the_same_geometry():
    """
    Fixed-time and actuated must differ ONLY in the timing rule. If the two
    networks disagreed about lanes or connections, the baselines would be
    driving different junctions and the comparison would be meaningless.
    """
    static = paths.net_file(N)
    actuated = paths.net_file(N, actuated=True)
    if not (static.exists() and actuated.exists()):
        pytest.skip("corridor not built")

    def movements(path):
        return sorted(
            (c.get("tl"), int(c.get("linkIndex")), c.get("from"), c.get("to"),
             c.get("fromLane"), c.get("dir"))
            for c in ET.parse(path).getroot().iter("connection")
            if c.get("tl") is not None)

    assert movements(static) == movements(actuated)
