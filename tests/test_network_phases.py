"""
Tests for the signal program built in networks/build_network.py.

WHY THESE TESTS EXIST
---------------------
The four phase strings ARE the safety design of this intersection. A phase is
one character per controlled movement, and netconvert - not us - decides which
movement each position refers to. If that mapping shifted between SUMO
versions, or if a phase were built from the wrong set of turn directions, the
simulation would happily run an intersection that gives green to two conflicting
movements at once. Nothing would crash; the numbers would simply be measured on
a junction that could not exist.

These tests pin the mapping against the committed network and assert the
invariants that make the phases safe, rather than only checking that a string
of the right length came out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from networks.build_network import (  # noqa: E402
    EAST_WEST,
    NET_FILE,
    NORTH_SOUTH,
    PHASE_NAMES,
    build_phase_states,
    phase_durations,
    read_controlled_links,
    yellow_state,
)

# The plan compiled into the committed network. Recorded here so that a change
# in SUMO's link ordering, or an edit to the phase construction, shows up as a
# failed test rather than as a silently different experiment.
EXPECTED_PHASES = [
    "GGrrrrGGrrrr",   # NS-through (plus right turns)
    "rrGrrrrrGrrr",   # NS-left
    "rrrGGrrrrGGr",   # EW-through (plus right turns)
    "rrrrrGrrrrrG",   # EW-left
]


@pytest.fixture(scope="module")
def links():
    if not NET_FILE.exists():
        pytest.skip("network not built - run: python networks/build_network.py")
    return read_controlled_links(NET_FILE)


@pytest.fixture(scope="module")
def phases(links):
    return build_phase_states(links)


# ---------------------------------------------------------------------------
# The network itself
# ---------------------------------------------------------------------------
def test_the_junction_has_twelve_controlled_movements(links):
    """Four approaches, three turns each, no U-turns (--no-turnarounds)."""
    assert len(links) == 12
    assert sorted(links) == list(range(12))


def test_every_approach_appears(links):
    edges = {edge for edge, _ in links.values()}
    assert edges == NORTH_SOUTH | EAST_WEST


def test_each_approach_has_one_movement_of_each_direction(links):
    for approach in NORTH_SOUTH | EAST_WEST:
        dirs = sorted(d for edge, d in links.values() if edge == approach)
        assert dirs == ["l", "r", "s"], (approach, dirs)


# ---------------------------------------------------------------------------
# The phase strings
# ---------------------------------------------------------------------------
def test_phase_strings_match_the_committed_plan(phases):
    assert phases == EXPECTED_PHASES


def test_there_are_four_phases_with_names_and_durations():
    assert len(PHASE_NAMES) == 4
    assert len(phase_durations()) == 4
    assert all(d > 0 for d in phase_durations())


def test_every_phase_is_one_character_per_movement(phases, links):
    assert all(len(p) == len(links) for p in phases)
    assert all(set(p) <= {"G", "r"} for p in phases)


def test_no_two_phases_give_green_to_the_same_movement(phases):
    """
    THE SAFETY INVARIANT. The four phases are meant to be mutually exclusive,
    which is what makes each of them internally conflict-free and lets the
    yellow phase be derived mechanically. If a movement were green in two
    phases, switching between them would drop it straight from green to green
    with no amber.
    """
    for i, first in enumerate(phases):
        for second in phases[i + 1:]:
            overlap = [k for k, (a, b) in enumerate(zip(first, second))
                       if a == "G" and b == "G"]
            assert not overlap, (first, second, overlap)


def test_every_movement_is_served_by_exactly_one_phase(phases, links):
    """
    No movement may be permanently red - that would starve an entire turn - and
    none may be served twice, per the exclusivity invariant above.
    """
    for index in links:
        serving = [n for n, p in zip(PHASE_NAMES, phases) if p[index] == "G"]
        assert len(serving) == 1, (index, links[index], serving)


def test_through_phases_also_carry_the_right_turn(phases, links):
    """
    Right turns run with the through movement rather than getting a phase of
    their own; they do not conflict with it. This is why the through phases
    show two greens per approach and the left phases only one.
    """
    ns_through = phases[0]
    greens = {links[i] for i, c in enumerate(ns_through) if c == "G"}
    assert greens == {(a, d) for a in NORTH_SOUTH for d in ("s", "r")}


def test_left_phases_are_protected_and_carry_nothing_else(phases, links):
    for phase, approaches in ((phases[1], NORTH_SOUTH), (phases[3], EAST_WEST)):
        greens = {links[i] for i, c in enumerate(phase) if c == "G"}
        assert greens == {(a, "l") for a in approaches}


def test_north_south_and_east_west_phases_never_share_a_movement(phases):
    ns = {i for p in (phases[0], phases[1]) for i, c in enumerate(p) if c == "G"}
    ew = {i for p in (phases[2], phases[3]) for i, c in enumerate(p) if c == "G"}
    assert not (ns & ew)


# ---------------------------------------------------------------------------
# yellow_state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("green", EXPECTED_PHASES)
def test_amber_replaces_green_and_leaves_red_alone(green):
    amber = yellow_state(green)
    assert len(amber) == len(green)
    assert "G" not in amber
    for g, y in zip(green, amber):
        assert y == ("y" if g == "G" else "r")


@pytest.mark.parametrize("green", EXPECTED_PHASES)
def test_amber_serves_exactly_the_movements_its_green_did(green):
    amber = yellow_state(green)
    assert [i for i, c in enumerate(green) if c == "G"] == \
           [i for i, c in enumerate(amber) if c == "y"]


def test_an_all_red_phase_stays_all_red():
    assert yellow_state("rrrr") == "rrrr"
