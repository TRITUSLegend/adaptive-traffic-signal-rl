"""
Tests for the Webster-tuned fixed-time plan (baseline/fixed_time.py).

WHY THESE TESTS EXIST
---------------------
This is the part of the project that has already shipped a real bug. The first
implementation used Webster's closed-form cycle length, which assumes green can
be divided freely in proportion to demand. It cannot here: every phase needs a
10 s minimum green, and two of the four phases serve light left-turn demand.
Clamping those two up to the minimum inflated the actual cycle while the
through phases kept the green they had been allocated for the shorter one, so
the through movement's share of the cycle FELL and the "tuned" plan came out
worse than the untuned one it was supposed to improve on.

That bug produced a full results table before anyone noticed the number looked
wrong. The invariant it violated - that the greens must fit inside the cycle
they were allocated for - is asserted directly below, so it cannot come back
silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline.fixed_time import (  # noqa: E402
    MAX_CYCLE,
    MIN_CYCLE,
    MIN_GREEN,
    SATURATION_FLOW,
    allocate_greens,
    flow_ratios,
    webster_delay,
    webster_green_times,
)
from eval.simulation import MIN_GREEN as SIM_MIN_GREEN  # noqa: E402
from eval.simulation import YELLOW_TIME  # noqa: E402
from networks.generate_routes import SCENARIOS  # noqa: E402

N_PHASES = 4
LOST_TIME = N_PHASES * YELLOW_TIME

# A few representative flow-ratio vectors: symmetric, lopsided, and one where
# the left-turn phases are so light they must be pinned to the minimum.
RATIO_CASES = [
    [0.18, 0.04, 0.18, 0.04],   # balanced-ish
    [0.27, 0.07, 0.13, 0.03],   # ns_heavy-ish
    [0.30, 0.01, 0.30, 0.01],   # both left phases far below their share
    [0.10, 0.10, 0.10, 0.10],   # perfectly even
]


# ---------------------------------------------------------------------------
# The constant that keeps the comparison fair (D5)
# ---------------------------------------------------------------------------
def test_min_green_is_not_defined_twice():
    """
    The baseline must use the same minimum green as the simulation layer.

    Both controllers being held to identical constraints is the entire basis of
    the report's fairness claim. This asserts there is one definition, not two
    that happen to agree today.
    """
    assert MIN_GREEN is SIM_MIN_GREEN


# ---------------------------------------------------------------------------
# allocate_greens
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ratios", RATIO_CASES)
@pytest.mark.parametrize("cycle", [56, 60, 75, 90, 104, 120])
def test_allocated_greens_fit_inside_their_cycle(cycle, ratios):
    """
    THE REGRESSION TEST. Greens plus lost time must equal the cycle exactly.

    This is the invariant the original closed-form implementation broke: after
    clamping the light phases up to the minimum, the greens summed to more than
    the cycle had room for, so the plan that actually ran had a longer cycle
    than the one the delay calculation had been performed on.
    """
    greens = allocate_greens(float(cycle), ratios)
    assert greens is not None
    assert sum(greens) + LOST_TIME == pytest.approx(float(cycle))


@pytest.mark.parametrize("ratios", RATIO_CASES)
@pytest.mark.parametrize("cycle", [56, 60, 75, 90, 104, 120])
def test_every_phase_clears_the_minimum_green(cycle, ratios):
    greens = allocate_greens(float(cycle), ratios)
    assert greens is not None
    assert all(g >= MIN_GREEN - 1e-9 for g in greens), greens


def test_returns_none_when_the_cycle_cannot_fit_every_minimum():
    """
    Below 4x10 s of green plus 16 s of amber there is no legal plan at all.

    Returning None rather than an illegal plan is what lets the search in
    webster_green_times skip infeasible cycles instead of quietly emitting one
    that violates the minimum green.
    """
    shortest_legal = N_PHASES * MIN_GREEN + LOST_TIME  # 56 s
    assert allocate_greens(float(shortest_legal - 1), RATIO_CASES[0]) is None
    assert allocate_greens(float(shortest_legal), RATIO_CASES[0]) is not None
    assert allocate_greens(float(MIN_CYCLE), RATIO_CASES[0]) is None  # 40 s


def test_at_the_shortest_legal_cycle_every_phase_gets_exactly_the_minimum():
    greens = allocate_greens(float(N_PHASES * MIN_GREEN + LOST_TIME),
                             RATIO_CASES[1])
    assert greens == pytest.approx([MIN_GREEN] * N_PHASES)


def test_busier_phases_get_more_green_when_the_floor_is_not_binding():
    """With room to spare, green is shared in proportion to demand."""
    ratios = [0.30, 0.10, 0.15, 0.05]
    greens = allocate_greens(120.0, ratios)
    assert greens is not None
    assert greens[0] > greens[2] > greens[1] > greens[3]


def test_pinning_one_phase_does_not_starve_another_below_the_floor():
    """
    Pinning is iterative: raising one phase to the floor reduces what is left
    for the others and can push a second below it. One pass is not enough, so
    this drives a case where two phases must be pinned in turn.
    """
    greens = allocate_greens(60.0, [0.48, 0.005, 0.48, 0.005])
    assert greens is not None
    assert all(g >= MIN_GREEN - 1e-9 for g in greens), greens
    assert sum(greens) + LOST_TIME == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# webster_delay
# ---------------------------------------------------------------------------
def test_no_demand_means_no_delay():
    assert webster_delay(90.0, 30.0, 0.0) == 0.0
    assert webster_delay(90.0, 30.0, -5.0) == 0.0


def test_oversaturation_returns_the_unusable_sentinel():
    """
    Average delay is not finite once demand exceeds capacity, so the function
    returns a huge ordered value instead. The search relies on this to reject
    infeasible plans rather than crashing or picking one.
    """
    # 10 s of green in a 100 s cycle gives a capacity of 180 veh/h.
    assert webster_delay(100.0, 10.0, 1000.0) >= 1e6


def test_more_green_lowers_delay_at_constant_demand():
    flow = 500.0
    delays = [webster_delay(90.0, g, flow) for g in (20.0, 30.0, 40.0, 50.0)]
    assert delays == sorted(delays, reverse=True)
    assert all(d > 0 for d in delays)


def test_delay_grows_as_demand_approaches_capacity():
    green, cycle = 40.0, 90.0
    capacity = SATURATION_FLOW * green / cycle
    delays = [webster_delay(cycle, green, capacity * s)
              for s in (0.3, 0.6, 0.9)]
    assert delays == sorted(delays)


# ---------------------------------------------------------------------------
# webster_green_times - the whole search, per scenario
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_scenario_produces_a_legal_plan(scenario):
    greens = webster_green_times(scenario)
    assert len(greens) == N_PHASES
    assert all(g >= MIN_GREEN for g in greens), (scenario, greens)
    cycle = sum(greens) + LOST_TIME
    # Rounding each green to a whole second can push the realised cycle a little
    # past the search bound, but not by more than one second per phase.
    assert MIN_CYCLE <= cycle <= MAX_CYCLE + N_PHASES


@pytest.mark.parametrize("scenario", ["ns_heavy", "peak", "surge", "light"])
def test_the_busy_direction_gets_more_green_than_the_quiet_one(scenario):
    """
    On every lopsided scenario, north-south carries twice east-west, so a tuned
    plan must give the north-south through phase more green. If it does not,
    the tuning is not doing the one thing it exists to do.
    """
    ns_through, _, ew_through, _ = webster_green_times(scenario)
    assert ns_through > ew_through, (scenario, ns_through, ew_through)


def test_symmetric_demand_gets_a_symmetric_plan():
    """
    The case the original bug got wrong.

    With equal demand on all four approaches the tuned plan should match the
    untuned one's symmetry: north-south and east-west must receive identical
    green. The broken implementation returned an asymmetric plan here and
    scored worse than doing nothing.
    """
    ns_through, ns_left, ew_through, ew_left = webster_green_times("balanced")
    assert ns_through == ew_through
    assert ns_left == ew_left


def test_ew_heavy_is_the_mirror_image_of_ns_heavy():
    """Mirroring the demand must mirror the plan, not change its shape."""
    ns = webster_green_times("ns_heavy")
    ew = webster_green_times("ew_heavy")
    assert ns == [ew[2], ew[3], ew[0], ew[1]]


# ---------------------------------------------------------------------------
# flow_ratios
# ---------------------------------------------------------------------------
def test_flow_ratios_are_fractions_of_saturation_flow():
    ratios = flow_ratios("ns_heavy")
    assert len(ratios) == N_PHASES
    assert all(0.0 < r < 1.0 for r in ratios)
    # ns_heavy: 600 veh/h per NS approach, 80% of it straight+right, against a
    # saturation flow of 1800 veh/h on the one kerb lane.
    assert ratios[0] == pytest.approx(600 * 0.8 / SATURATION_FLOW)
    # The through phase carries four times the left phase's 20% share.
    assert ratios[0] == pytest.approx(ratios[1] * 4.0)


def test_the_critical_movement_sizes_the_phase():
    """
    Opposing approaches run together, so a phase is sized by the busier of the
    two - not by their sum and not by their average.
    """
    ratios = flow_ratios("ns_heavy")
    ns_flow = max(SCENARIOS["ns_heavy"]["flows"][a] for a in ("N2C", "S2C"))
    assert ratios[0] == pytest.approx(ns_flow * 0.8 / SATURATION_FLOW)
