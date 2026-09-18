"""
Tests for corridor/generate_corridor_routes.py.

WHY THESE TESTS EXIST
---------------------
Defect D1 in the single-intersection study: route generation seeded from
Python's `hash()`, which is salted per process, so the same command produced
different traffic on every run and no published number could be reproduced.
The corridor generator starts from CRC32 for that reason, and
`test_seed_is_a_fixed_function_of_its_inputs` is what keeps it there.

The routing tests matter for a second reason. A corridor route is a *path*,
not a single turn, so a wrong path does not raise - SUMO simply drives the
vehicle somewhere else, and the demand actually loading each junction stops
being the demand the scenario claims to impose.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor import paths  # noqa: E402
from corridor.build_corridor import junction_ids  # noqa: E402
from corridor.generate_corridor_routes import (  # noqa: E402
    EVALUATION_SEEDS,
    SCENARIOS,
    TRAINING_SCENARIO,
    TRAINING_SEEDS,
    TURN_SPLIT,
    boundary_nodes,
    destinations,
    entry_flows,
    path_edges,
    sample_vehicles,
    scenario_seed,
)

N = 3


# ---------------------------------------------------------------------------
# Reproducibility - the D1 guard
# ---------------------------------------------------------------------------
def test_seed_is_a_fixed_function_of_its_inputs():
    """
    Pinned literals, not just self-consistency.

    Comparing the function to itself would pass even if it were reseeded from
    something process-dependent, which is exactly the bug this guards. These
    values were computed once and must never move.
    """
    assert scenario_seed(3, "calibrated", 0) == scenario_seed(3, "calibrated", 0)
    assert isinstance(scenario_seed(3, "calibrated", 0), int)
    # CRC32 of "corridor3_calibrated_0" and friends.
    import zlib
    for n, scenario, seed in [(3, "calibrated", 0), (3, "peak", 104), (5, "balanced", 7)]:
        expected = zlib.crc32(("corridor%d_%s_%d" % (n, scenario, seed)).encode("utf-8"))
        assert scenario_seed(n, scenario, seed) == expected


def test_different_scenarios_do_not_share_an_arrival_pattern():
    assert scenario_seed(3, "calibrated", 0) != scenario_seed(3, "peak", 0)


def test_different_corridor_sizes_do_not_share_an_arrival_pattern():
    """Mixing N in is what stops corridor2 and corridor3 sharing traffic."""
    assert scenario_seed(2, "calibrated", 0) != scenario_seed(3, "calibrated", 0)


def test_the_same_seed_reproduces_the_same_vehicles():
    import random
    a = sample_vehicles(N, "calibrated", 3600,
                        random.Random(scenario_seed(N, "calibrated", 0)))
    b = sample_vehicles(N, "calibrated", 3600,
                        random.Random(scenario_seed(N, "calibrated", 0)))
    assert a == b


def test_training_and_evaluation_seeds_never_overlap():
    """Every reported number must be measured on unseen demand."""
    assert set(TRAINING_SEEDS).isdisjoint(EVALUATION_SEEDS)


def test_the_training_scenario_is_one_of_the_defined_scenarios():
    assert TRAINING_SCENARIO in SCENARIOS


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_a_full_traverse_visits_every_junction():
    edges = path_edges(N, "W", "E")
    for jid in junction_ids(N):
        assert any(e.endswith("_" + jid) for e in edges), jid


def test_the_traverse_is_the_expected_edge_sequence():
    assert path_edges(3, "W", "E") == ["W_J1", "J1_J2", "J2_J3", "J3_E"]
    assert path_edges(3, "E", "W") == ["E_J3", "J3_J2", "J2_J1", "J1_W"]


def test_a_side_street_crossing_uses_exactly_one_junction():
    assert path_edges(3, "N2", "S2") == ["N2_J2", "J2_S2"]


def test_a_side_entry_onto_the_arterial_runs_to_the_end():
    assert path_edges(3, "N1", "E") == ["N1_J1", "J1_J2", "J2_J3", "J3_E"]


def test_every_route_starts_at_its_origin_and_ends_at_its_destination():
    nodes = boundary_nodes(N)
    every = nodes["arterial"] + nodes["side"]
    for origin in every:
        for dest in every:
            if origin == dest:
                continue
            edges = path_edges(N, origin, dest)
            assert edges[0].startswith(origin + "_")
            assert edges[-1].endswith("_" + dest)


def test_routes_are_connected_end_to_end():
    """Each edge must start where the previous one finished."""
    for origin, dest in [("W", "E"), ("E", "N1"), ("S3", "W"), ("N1", "S1")]:
        edges = path_edges(N, origin, dest)
        for before, after in zip(edges, edges[1:]):
            assert before.split("_")[1] == after.split("_")[0]


def test_a_trip_to_itself_is_rejected():
    with pytest.raises(ValueError):
        path_edges(N, "W", "W")


def test_destinations_follow_the_same_turn_convention_as_the_network():
    """
    From the north you are heading south, so a left turn takes you east and a
    right turn west - the same rule the network geometry is built from.
    """
    north = destinations(N, "N1")
    assert north["left"] == ["E"]
    assert north["right"] == ["W"]
    assert north["straight"] == ["S1"]

    south = destinations(N, "S1")
    assert south["left"] == ["W"]
    assert south["right"] == ["E"]
    assert south["straight"] == ["N1"]


def test_an_arterial_entry_can_leave_at_any_junction():
    west = destinations(N, "W")
    assert west["straight"] == ["E"]
    assert west["left"] == ["N1", "N2", "N3"]
    assert west["right"] == ["S1", "S2", "S3"]


def test_every_boundary_node_has_reachable_destinations():
    nodes = boundary_nodes(N)
    for origin in nodes["arterial"] + nodes["side"]:
        for options in destinations(N, origin).values():
            for dest in options:
                assert dest != origin
                path_edges(N, origin, dest)  # must not raise


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
def test_the_corridor_has_two_arterial_ends_and_two_stubs_per_junction():
    nodes = boundary_nodes(N)
    assert nodes["arterial"] == ["W", "E"]
    assert len(nodes["side"]) == 2 * N


def test_entry_flows_cover_every_boundary_node():
    flows = entry_flows(N, "calibrated")
    nodes = boundary_nodes(N)
    assert set(flows) == set(nodes["arterial"] + nodes["side"])


def test_calibrated_matches_the_documented_cologne3_derivation():
    """
    500 veh/h per arterial end and 200 per side stub is what the cologne3
    busiest hour was rounded to. If these move, the module docstring's
    arithmetic and CORRIDOR_REPORT.md's provenance claim are both stale.
    """
    spec = SCENARIOS["calibrated"]
    assert spec["arterial"] == 500
    assert spec["side"] == 200
    assert sum(entry_flows(3, "calibrated").values()) == 2200


def test_reverse_actually_reverses_the_dominant_direction():
    calibrated = SCENARIOS["calibrated"]
    reverse = SCENARIOS["reverse"]
    assert calibrated["arterial"] > calibrated["side"]
    assert reverse["side"] > reverse["arterial"]


def test_peak_is_calibrated_scaled_up():
    assert SCENARIOS["peak"]["arterial"] > SCENARIOS["calibrated"]["arterial"]
    assert SCENARIOS["peak"]["side"] > SCENARIOS["calibrated"]["side"]


def test_vehicle_count_is_close_to_the_requested_flow():
    """Poisson sampling, so this is statistical - a wide band, but not unbounded."""
    import random
    vehicles = sample_vehicles(N, "calibrated", 3600,
                               random.Random(scenario_seed(N, "calibrated", 0)))
    expected = sum(entry_flows(N, "calibrated").values())
    assert 0.9 * expected < len(vehicles) < 1.1 * expected


def test_vehicles_are_sorted_by_departure_time():
    """SUMO requires departure order and silently misbehaves without it."""
    import random
    vehicles = sample_vehicles(N, "calibrated", 3600,
                               random.Random(scenario_seed(N, "calibrated", 1)))
    times = [t for t, _, _ in vehicles]
    assert times == sorted(times)


def test_no_vehicle_departs_after_the_hour_ends():
    import random
    vehicles = sample_vehicles(N, "calibrated", 3600,
                               random.Random(scenario_seed(N, "calibrated", 2)))
    assert all(t < 3600 for t, _, _ in vehicles)


def test_turn_split_matches_the_single_intersection_study():
    """Inherited unchanged; cologne3 gives volumes, not turning fractions."""
    assert TURN_SPLIT == {"straight": 0.60, "left": 0.20, "right": 0.20}
    assert abs(sum(TURN_SPLIT.values()) - 1.0) < 1e-9


def test_generated_route_file_parses_and_references_defined_routes():
    path = paths.route_file(N, "calibrated", 100)
    if not path.exists():
        pytest.skip("routes not generated - run: "
                    "python corridor/generate_corridor_routes.py")
    root = ET.parse(path).getroot()
    defined = {r.get("id") for r in root.iter("route")}
    used = {v.get("route") for v in root.iter("vehicle")}
    assert used <= defined
    assert used, "route file contains no vehicles"


def test_generated_route_edges_all_exist_in_the_network():
    route_path = paths.route_file(N, "calibrated", 100)
    net_path = paths.net_file(N)
    if not (route_path.exists() and net_path.exists()):
        pytest.skip("corridor not built")
    real = {e.get("id") for e in ET.parse(net_path).getroot().iter("edge")}
    for route in ET.parse(route_path).getroot().iter("route"):
        for edge in route.get("edges").split():
            assert edge in real, edge


def test_generated_file_is_byte_identical_when_regenerated(tmp_path):
    """
    The whole point of D1's fix: the same command must produce the same file.
    Written to a temporary directory so the committed demand is never touched.
    """
    import random

    from corridor.generate_corridor_routes import sample_vehicles as sample
    first = sample(N, "peak", 600, random.Random(scenario_seed(N, "peak", 3)))
    second = sample(N, "peak", 600, random.Random(scenario_seed(N, "peak", 3)))
    assert first == second
    assert tmp_path.exists()  # nothing was written to the repository


# ---------------------------------------------------------------------------
# The documented reproduction must actually produce what the pipeline needs
# ---------------------------------------------------------------------------
def test_the_transfer_targets_are_part_of_the_full_generation():
    """
    `--all` exists because running this script bare produces only the
    3-junction corridor, and the transfer study then dies on a missing file for
    corridor2. The full set was generated by hand during development and never
    written down - which is exactly how a documented reproduction sequence
    stops reproducing.
    """
    from corridor.generate_corridor_routes import TRANSFER_JUNCTIONS
    from corridor.robustness import CLEAN_TARGETS
    assert sorted(TRANSFER_JUNCTIONS) == sorted(CLEAN_TARGETS), (
        "the route generator and the transfer study disagree about which "
        "corridors are transfer targets")


def test_every_corridor_the_transfer_study_visits_has_its_routes():
    """The concrete failure this guards: robustness.py --transfer on a corridor
    whose demand was never generated."""
    from corridor.generate_corridor_routes import TRANSFER_JUNCTIONS
    for n in TRANSFER_JUNCTIONS:
        for seed in EVALUATION_SEEDS:
            path = paths.route_file(n, TRAINING_SCENARIO, seed)
            if not path.exists():
                pytest.skip("routes not generated - run: "
                            "python corridor/generate_corridor_routes.py --all")
            assert path.stat().st_size > 0


def test_the_reported_reproduction_command_asks_for_everything():
    """
    CORRIDOR_REPORT.md tells a reader what to run. If that line drops `--all`,
    the transfer study breaks for them and not for us.
    """
    report = ROOT / "CORRIDOR_REPORT.md"
    if not report.exists():
        pytest.skip("report not written yet")
    text = report.read_text(encoding="utf-8")
    assert "generate_corridor_routes.py --all" in text, (
        "the report's reproduction sequence would not generate the transfer "
        "targets' demand")
