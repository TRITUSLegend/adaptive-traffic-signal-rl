"""
Tests for corridor/phase_service.py - measuring starvation from the traces.

WHY THESE TESTS EXIST
---------------------
This module answers a question no aggregate metric can: whether a controller
quietly stops serving a movement. Addendum E of the single-intersection study
found exactly that - 245-340 s unserved against a 200 s bound - and found it
only by reading the signal's own behaviour, because a starved approach carries
too few vehicles to move the mean waiting time.

A number like "longest a phase went unserved" is believed on sight and is easy
to get subtly wrong: counting amber as service, forgetting that a phase not yet
seen has not been waiting, or resetting the wrong clock. Each of those produces
a smaller, more reassuring number than the truth. So the arithmetic is pinned
on hand-computable traces here rather than trusted.

All fixtures are written to pytest's tmp_path, never to a real output path.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor.phase_service import analyse_trace, signal_columns  # noqa: E402
from eval.simulation import DELTA_TIME  # noqa: E402

# The corridor's four green phases and the amber that follows phase A.
A = "GGrrrrGGrrrr"
B = "rrGrrrrrGrrr"
C = "rrrGGrrrrGGr"
D = "rrrrrGrrrrrG"
AMBER = "yyrrrryyrrrr"


def write_trace(tmp_path: Path, states, junction: str = "J1") -> Path:
    """One trace file whose junction shows `states`, one per decision step."""
    path = tmp_path / "trace.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sim_time", "queue_veh", "signal_%s" % junction])
        for i, state in enumerate(states):
            writer.writerow([i * DELTA_TIME, 0.0, state])
    return path


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------
def test_signal_columns_are_found_by_prefix():
    header = ["sim_time", "queue_veh", "signal_J1", "signal_J2", "reward"]
    assert signal_columns(header) == ["signal_J1", "signal_J2"]


def test_a_trace_without_signal_columns_yields_nothing(tmp_path):
    path = tmp_path / "bare.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sim_time", "queue_veh"])
        writer.writerow([0, 1.0])
    assert analyse_trace(path) == {}


def test_a_missing_trace_is_not_an_error(tmp_path):
    assert analyse_trace(tmp_path / "absent.csv") == {}


def test_every_junction_in_the_trace_is_reported(tmp_path):
    path = tmp_path / "two.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sim_time", "signal_J1", "signal_J2"])
        for i in range(4):
            writer.writerow([i * DELTA_TIME, A, C])
    assert sorted(analyse_trace(path)) == ["J1", "J2"]


# ---------------------------------------------------------------------------
# Longest continuous green
# ---------------------------------------------------------------------------
def test_longest_green_counts_consecutive_identical_phases(tmp_path):
    """Four steps of A at 5 s each is 20 s of green."""
    result = analyse_trace(write_trace(tmp_path, [A, A, A, A]))["J1"]
    assert result["longest_green_s"] == 4 * DELTA_TIME


def test_longest_green_takes_the_maximum_not_the_last_run(tmp_path):
    states = [A, A, A, A, C, A]        # a 20 s run, then a 5 s one
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    assert result["longest_green_s"] == 4 * DELTA_TIME


def test_amber_breaks_a_green_run(tmp_path):
    """
    Amber is not green. A run interrupted by amber is two runs, and treating it
    as one would overstate the longest green - the wrong direction for a
    measure whose job is to catch a signal holding on too long.
    """
    states = [A, A, AMBER, A, A]
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    assert result["longest_green_s"] == 2 * DELTA_TIME


# ---------------------------------------------------------------------------
# Longest a phase went unserved - the starvation measure
# ---------------------------------------------------------------------------
def test_a_phase_never_shown_is_not_counted_as_waiting(tmp_path):
    """
    A phase that has not appeared yet has no clock. Starting one at time zero
    for all four would report starvation on any episode that simply had not got
    around to a phase yet, which is a different claim.
    """
    result = analyse_trace(write_trace(tmp_path, [A, A, A]))["J1"]
    assert result["phases_seen"] == 1
    assert result["longest_unserved_s"] == 0


def test_a_phase_waits_while_another_is_served(tmp_path):
    """A is shown, then C for three steps: A has waited 15 s."""
    states = [A, C, C, C]
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    assert result["phases_seen"] == 2
    assert result["longest_unserved_s"] == 3 * DELTA_TIME


def test_serving_a_phase_resets_its_clock(tmp_path):
    states = [A, C, C, A, C, C]
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    # A waits 10 s, is served, waits 10 s again - never more than 10.
    assert result["longest_unserved_s"] == 2 * DELTA_TIME


def test_amber_counts_as_waiting_for_everyone(tmp_path):
    """
    During amber no phase has right of way, so nobody is being served and
    everybody's clock advances. Treating amber as service would shorten every
    reported wait.
    """
    states = [A, C, AMBER, C]
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    assert result["longest_unserved_s"] == 3 * DELTA_TIME     # A waits 3 steps


def test_a_starved_phase_shows_up_as_a_long_wait(tmp_path):
    """The case this module exists for."""
    states = [A, B] + [C] * 60          # 300 s in which neither A nor B runs
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    assert result["longest_unserved_s"] >= 300


def test_a_phase_that_never_appears_shows_as_a_missing_phase(tmp_path):
    """
    The blunt measure. Four phases exist; if only three are ever shown, the
    fourth was starved outright and no waiting-time figure would reveal it.
    """
    states = ([A, B, C] * 20)
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    assert result["phases_seen"] == 3


def test_all_four_phases_cycling_is_reported_as_healthy(tmp_path):
    states = ([A, B, C, D] * 10)
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    assert result["phases_seen"] == 4
    assert result["longest_unserved_s"] == 3 * DELTA_TIME


def test_an_empty_state_string_is_skipped(tmp_path):
    """A junction whose state could not be read must not corrupt the counts."""
    states = [A, "", A, A]
    result = analyse_trace(write_trace(tmp_path, states))["J1"]
    assert result["phases_seen"] == 1
    assert result["longest_green_s"] >= 2 * DELTA_TIME
