"""
The corridor's output filenames must come from one place and be unambiguous.

WHY THIS TEST EXISTS
--------------------
Defect F2 in the single-intersection study: the trace filename was built in
three places, two of them disagreed, and because "no trace file" is a
legitimate state the mismatch did not raise. It silently dropped rows from the
one table that existed to show them, and Addendum E would have published with
its key controllers missing. `eval/paths.py` fixed that by becoming the single
naming authority, and `tests/test_output_naming.py` pins it.

`corridor/paths.py` is the corridor's equivalent and has strictly more names to
get wrong - controllers, scenarios, seeds, failure rates, modes and corridor
sizes all appear in filenames. The audit found that tests USED it as a helper
but nothing ASSERTED its behaviour, so the F2 failure mode was fully available
and unguarded on the newer half of the project. This module closes that gap.

The dangerous property is not that a name looks wrong. It is that two callers
can derive *different* names for the same artifact, or the *same* name for
different artifacts - and both fail silently, as a missing file rather than an
error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor import paths  # noqa: E402


# ---------------------------------------------------------------------------
# One authority: the writer and the readers must use the same function object
# ---------------------------------------------------------------------------
def test_every_module_uses_the_same_trace_naming_function():
    """
    F2 was three implementations, not three spellings. Identity of the function
    object is the check that actually catches a second implementation
    appearing; comparing output strings would pass right up until the day the
    copies diverged.
    """
    import corridor.evaluate_corridor as evaluate
    import corridor.make_corridor_tables as tables
    import corridor.phase_service as service

    assert evaluate.paths.trace_file is paths.trace_file
    assert tables.paths.trace_file is paths.trace_file
    assert service.paths.trace_file is paths.trace_file


def test_the_sweeps_and_the_tables_agree_on_where_results_live():
    """The writer and the reader of every results CSV are the same constant."""
    import corridor.evaluate_corridor as evaluate
    import corridor.make_corridor_tables as tables
    import corridor.robustness as robustness

    assert evaluate.paths.EVALUATION_RAW is paths.EVALUATION_RAW
    assert tables.paths.EVALUATION_RAW is paths.EVALUATION_RAW
    assert robustness.paths.ROBUSTNESS_RAW is paths.ROBUSTNESS_RAW
    assert tables.paths.ROBUSTNESS_RAW is paths.ROBUSTNESS_RAW
    assert robustness.paths.TRANSFER_RAW is paths.TRANSFER_RAW
    assert tables.paths.TRANSFER_RAW is paths.TRANSFER_RAW


# ---------------------------------------------------------------------------
# Distinct artifacts must never collide
# ---------------------------------------------------------------------------
def test_different_controllers_get_different_trace_files():
    names = [paths.trace_file(c, "calibrated", 3).name
             for c in ("fixed-time", "actuated", "greedy-queue",
                       "idqn", "ippo", "ma2c", "ma2c-sampled")]
    assert len(set(names)) == len(names), "two controllers share a trace name"


def test_a_plus_safe_controller_does_not_collide_with_its_base():
    """
    The single-intersection study keeps `+safe` in filenames deliberately. If
    it were stripped or normalised to an underscore, a wrapped run would
    overwrite the unwrapped one and the comparison that measures the wrapper's
    cost would silently compare a file with itself.
    """
    assert (paths.trace_file("idqn+safe", "calibrated", 3)
            != paths.trace_file("idqn", "calibrated", 3))
    assert (paths.trace_file("idqn+safe", "calibrated", 3)
            != paths.trace_file("idqn_safe", "calibrated", 3))


def test_different_scenarios_get_different_trace_files():
    names = [paths.trace_file("idqn", s, 3).name
             for s in ("calibrated", "balanced", "peak", "reverse")]
    assert len(set(names)) == len(names)


def test_different_corridor_sizes_get_different_trace_files():
    """
    The transfer study runs the same controller on 2, 3 and 5 junctions. If the
    size were dropped from the name, corridor5's trace would overwrite
    corridor3's and the transfer table would quietly describe one network
    three times.
    """
    names = [paths.trace_file("idqn", "calibrated", n).name for n in (2, 3, 5)]
    assert len(set(names)) == len(names)


def test_different_seeds_get_different_tripinfo_files():
    names = [paths.tripinfo_file("idqn", "calibrated", s, 3).name
             for s in range(100, 105)]
    assert len(set(names)) == len(names)


def test_tripinfo_separates_controller_scenario_seed_and_size():
    a = paths.tripinfo_file("idqn", "calibrated", 100, 3)
    for other in (paths.tripinfo_file("ippo", "calibrated", 100, 3),
                  paths.tripinfo_file("idqn", "peak", 100, 3),
                  paths.tripinfo_file("idqn", "calibrated", 101, 3),
                  paths.tripinfo_file("idqn", "calibrated", 100, 5)):
        assert a != other


def test_model_and_curve_names_distinguish_algorithm_and_tag():
    assert paths.model_file("idqn") != paths.model_file("ippo")
    assert paths.model_file("idqn") != paths.model_file("idqn", "probe")
    assert paths.training_curve("ma2c") != paths.training_curve("ma2c", "probe")


def test_ma2c_is_stored_as_torch_and_the_others_as_zip():
    """
    MA2C is hand-rolled PyTorch; the independent learners are SB3 archives.
    Loading one as the other fails loudly, but only if the suffix is right.
    """
    assert paths.model_file("ma2c").suffix == ".pt"
    assert paths.model_file("idqn").suffix == ".zip"
    assert paths.model_file("ippo").suffix == ".zip"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_corridor_name_is_derived_from_the_junction_count():
    assert paths.corridor_name(2) == "corridor2"
    assert paths.corridor_name(3) == "corridor3"
    assert paths.corridor_name(5) == "corridor5"


def test_a_corridor_needs_at_least_one_junction():
    with pytest.raises(ValueError):
        paths.corridor_name(0)


def test_the_actuated_network_is_a_separate_file_from_the_static_one():
    """
    Fixed-time and actuated differ only in timing rule, and they are two
    compiled files. One name for both would mean whichever was built last
    silently drove both baselines.
    """
    assert paths.net_file(3, actuated=False) != paths.net_file(3, actuated=True)
    assert "actuated" in paths.net_file(3, actuated=True).name


def test_every_output_path_stays_under_the_corridor_results_tree():
    """
    Structural separation from the single-intersection study. Nothing the
    corridor writes may land in results/logs/ or results/plots/, whatever the
    arguments.
    """
    single_logs = paths.ROOT / "results" / "logs"
    single_plots = paths.ROOT / "results" / "plots"
    produced = [
        paths.trace_file("idqn", "calibrated", 3),
        paths.tripinfo_file("idqn", "calibrated", 100, 3),
        paths.training_curve("idqn"),
        paths.EVALUATION_RAW, paths.ROBUSTNESS_RAW, paths.TRANSFER_RAW,
    ]
    for path in produced:
        assert single_logs not in path.parents
        assert single_plots not in path.parents
        assert paths.RESULTS_DIR in path.parents


def test_models_and_routes_stay_out_of_the_single_intersection_directories():
    forbidden = (paths.ROOT / "agent" / "models",
                 paths.ROOT / "networks" / "routes")
    produced = (paths.model_file("idqn"), paths.route_file(3, "calibrated", 100))
    for path in produced:
        for bad in forbidden:
            assert bad not in path.parents and path != bad


def test_names_are_stable_across_calls():
    """A path that changed between the writer and the reader would produce the
    F2 symptom exactly: a file that exists under a name nobody looks up."""
    for _ in range(3):
        assert (paths.trace_file("ma2c-sampled", "reverse", 3)
                == paths.trace_file("ma2c-sampled", "reverse", 3))
