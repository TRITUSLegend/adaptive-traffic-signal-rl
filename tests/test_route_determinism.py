"""
Tests for demand generation (networks/generate_routes.py).

WHY THESE TESTS EXIST
---------------------
The route files ARE the experiment's independent variable. Every controller is
compared on the identical list of vehicles, and the report's whole fairness
argument depends on "scenario + seed" naming exactly one hour of traffic.

The original implementation seeded its generator with Python's built-in
``hash()`` of a string. That is salted per interpreter process, so the same
command produced different traffic on every run: three consecutive runs of
``ns_heavy`` seed 100 gave 1759, 1759 and 1867 vehicles, all with different
arrival sequences. Nothing failed and nothing looked wrong - but any reader who
followed the README's instruction to regenerate the demand would have silently
replaced the traffic every published number was measured on.

The cross-process half of this cannot be tested from inside one interpreter, so
`test_seed_is_a_fixed_function_of_its_inputs` pins the seed to its literal
value. Any return to a process-dependent function fails it immediately.
"""

from __future__ import annotations

import random
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from networks.generate_routes import (  # noqa: E402
    EVALUATION_SEEDS,
    SCENARIOS,
    TRAINING_SCENARIO,
    TRAINING_SEEDS,
    TURN_SPLIT,
    sample_vehicles,
    scenario_seed,
    scenario_seeds,
    write_scenario,
)


# ---------------------------------------------------------------------------
# The seed itself
# ---------------------------------------------------------------------------
def test_seed_is_a_fixed_function_of_its_inputs():
    """
    THE REGRESSION TEST for the salted-hash bug.

    These literals are what CRC-32 returns for these strings - the same on every
    machine, every Python version and every run. Should the seeding ever go back
    to `hash()`, or to anything else process-dependent, this fails at once
    instead of quietly producing different traffic.
    """
    assert scenario_seed("ns_heavy", 100) == zlib.crc32(b"ns_heavy_100")
    assert scenario_seed("ns_heavy", 100) == 2717491218
    assert scenario_seed("balanced", 0) == zlib.crc32(b"balanced_0")


def test_the_scenario_name_is_mixed_into_the_seed():
    """
    Two scenarios sharing a seed number must not share an arrival pattern -
    otherwise `balanced` seed 100 and `peak` seed 100 would be the same hour
    scaled, and the scenarios would not be independent samples.
    """
    for seed in (0, 100, 104):
        seeds = {name: scenario_seed(name, seed) for name in SCENARIOS}
        assert len(set(seeds.values())) == len(SCENARIOS), seeds


def test_different_seeds_give_different_streams():
    assert scenario_seed("ns_heavy", 100) != scenario_seed("ns_heavy", 101)


# ---------------------------------------------------------------------------
# The generated traffic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_regenerating_gives_identical_traffic(scenario, tmp_path):
    """
    The property the whole thing exists for: same inputs, same vehicles.

    Two independent generators seeded from the same (scenario, seed) must
    produce byte-identical files.
    """
    first = write_scenario(scenario, 100, 3600, tmp_path / "a")
    second = write_scenario(scenario, 100, 3600, tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()


def test_sampling_is_reproducible_from_a_fresh_generator():
    flows = SCENARIOS["ns_heavy"]["flows"]
    seed = scenario_seed("ns_heavy", 100)
    first = sample_vehicles(flows, 3600, random.Random(seed))
    second = sample_vehicles(flows, 3600, random.Random(seed))
    assert first == second
    assert len(first) > 0


def test_vehicles_are_sorted_by_departure_time():
    """SUMO requires departure order and rejects a file that is out of order."""
    vehicles = sample_vehicles(SCENARIOS["peak"]["flows"], 3600,
                               random.Random(scenario_seed("peak", 100)))
    departures = [t for t, _, _ in vehicles]
    assert departures == sorted(departures)


def test_no_vehicle_departs_after_the_episode_ends():
    duration = 600
    vehicles = sample_vehicles(SCENARIOS["ns_heavy"]["flows"], duration,
                               random.Random(1))
    assert all(0.0 <= t < duration for t, _, _ in vehicles)


def test_arrival_volume_is_close_to_the_demanded_rate():
    """
    Poisson arrivals, so the count is random - but over an hour at 600 veh/h
    the mean is 600 with a standard deviation of about 24, and five standard
    deviations is a generous band that still catches a rate bug.
    """
    flows = {"N2C": 600}
    vehicles = sample_vehicles(flows, 3600, random.Random(7))
    assert 480 < len(vehicles) < 720


def test_turn_proportions_are_roughly_as_configured():
    vehicles = sample_vehicles({"N2C": 3000}, 3600, random.Random(11))
    total = len(vehicles)
    for turn, share in TURN_SPLIT.items():
        seen = sum(1 for _, _, t in vehicles if t == turn) / total
        assert abs(seen - share) < 0.05, (turn, seen, share)


# ---------------------------------------------------------------------------
# Which files the script owns
# ---------------------------------------------------------------------------
def test_the_training_scenario_gets_the_training_seeds():
    seeds = scenario_seeds(TRAINING_SCENARIO)
    assert set(TRAINING_SEEDS) <= set(seeds)
    assert set(EVALUATION_SEEDS) <= set(seeds)


@pytest.mark.parametrize(
    "scenario", [s for s in SCENARIOS if s != TRAINING_SCENARIO])
def test_other_scenarios_get_only_the_evaluation_seeds(scenario):
    assert scenario_seeds(scenario) == sorted(EVALUATION_SEEDS)


def test_training_and_evaluation_seeds_never_overlap():
    """
    The held-out-test-set guarantee. If these ever intersected, a good score
    could just mean the agent had memorised the hour it was tested on.
    """
    assert not set(TRAINING_SEEDS) & set(EVALUATION_SEEDS)


# ---------------------------------------------------------------------------
# The committed files must not be regenerable by accident
# ---------------------------------------------------------------------------
def test_the_committed_route_files_are_still_present():
    """
    The demand every published number was measured on. It predates the seeding
    fix and cannot be reproduced by the current generator, so losing it would
    mean losing the ability to re-run the reported evaluation. This is why the
    generator refuses to overwrite without --force.
    """
    routes = ROOT / "networks" / "routes"
    expected = sum(len(scenario_seeds(s)) for s in SCENARIOS)
    found = sorted(routes.glob("*.rou.xml"))
    assert len(found) == expected, "expected %d route files, found %d" % (
        expected, len(found))
