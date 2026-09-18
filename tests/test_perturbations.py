"""
Tests for corridor/perturbations.py - the detector-failure harness.

WHY THESE TESTS EXIST
---------------------
The robustness study is the part of this work the brief calls the novelty, and
its result is only meaningful if two properties hold exactly:

  1. **The perturbation is paired.** Every controller must lose the SAME
     detectors on the same hour. If the failed set depended on anything about
     the controller, the "degradation" being reported would be partly a
     difference in the fault, and no amount of averaging would separate them.
  2. **Only sensing is affected.** The phase one-hot and the minimum-green flag
     are the signal's own state, not measurements. Corrupting them would model
     a different fault, and the controller would be degraded in a way no
     detector failure causes.

Both would fail silently: the sweep would run, produce numbers, and answer a
question nobody asked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor.perturbations import (  # noqa: E402
    DENSITY_SLICE,
    MODES,
    N_LANES,
    PHASE_SLICE,
    QUEUE_SLICE,
    DetectorFailure,
    NoPerturbation,
    make_perturbation,
)

IDS = ["J1", "J2", "J3"]


class FakeEnv:
    def __init__(self, ts_ids=IDS):
        self.ts_ids = list(ts_ids)


def observation(value: float = 0.5) -> np.ndarray:
    """A 21-value observation with a distinctive value in every slot."""
    obs = np.full(21, value, dtype=np.float32)
    obs[PHASE_SLICE] = 1.0
    return obs


def observations(value: float = 0.5) -> dict:
    return {ts: observation(value) for ts in IDS}


# ---------------------------------------------------------------------------
# Pairing: the fairness guarantee
# ---------------------------------------------------------------------------
def test_the_same_episode_fails_the_same_detectors():
    a = DetectorFailure(0.25, "zero", scenario="calibrated", seed=100)
    b = DetectorFailure(0.25, "zero", scenario="calibrated", seed=100)
    a.on_episode_start(FakeEnv())
    b.on_episode_start(FakeEnv())
    assert a.failed == b.failed


def test_nothing_about_the_controller_enters_the_draw():
    """
    The failed set is a function of (scenario, seed, rate, mode) only. There is
    deliberately no controller argument to pass, so this test pins the
    signature as much as the behaviour.
    """
    import inspect
    params = set(inspect.signature(DetectorFailure.__init__).parameters)
    assert params == {"self", "rate", "mode", "scenario", "seed"}


def test_different_seeds_fail_different_detectors():
    a = DetectorFailure(0.25, "zero", scenario="calibrated", seed=100)
    b = DetectorFailure(0.25, "zero", scenario="calibrated", seed=101)
    a.on_episode_start(FakeEnv())
    b.on_episode_start(FakeEnv())
    assert a.failed != b.failed


def test_different_rates_are_distinguishable():
    a = DetectorFailure(0.25, "zero", scenario="calibrated", seed=100)
    b = DetectorFailure(0.50, "zero", scenario="calibrated", seed=100)
    a.on_episode_start(FakeEnv())
    b.on_episode_start(FakeEnv())
    assert a.failed != b.failed


def test_the_failed_set_is_held_for_the_whole_episode():
    """A dead detector does not recover halfway through the hour."""
    fail = DetectorFailure(0.25, "zero", scenario="calibrated", seed=100)
    fail.on_episode_start(FakeEnv())
    first = {k: list(v) for k, v in fail.failed.items()}
    for _ in range(10):
        fail.corrupt(observations())
    assert {k: list(v) for k, v in fail.failed.items()} == first


# ---------------------------------------------------------------------------
# How many, and where
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rate,expected", [(0.0, 0), (0.25, 2), (0.5, 4), (1.0, 8)])
def test_failure_rate_selects_that_fraction_of_each_junctions_lanes(rate, expected):
    fail = DetectorFailure(rate, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    for ts in IDS:
        assert len(fail.failed[ts]) == expected


def test_failures_are_spread_across_junctions_not_concentrated():
    """25% must mean a quarter of every junction, not a quarter of the corridor."""
    fail = DetectorFailure(0.25, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    assert set(fail.failed) == set(IDS)
    assert all(len(v) == 2 for v in fail.failed.values())


def test_a_lane_is_never_failed_twice():
    fail = DetectorFailure(0.5, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    for lanes in fail.failed.values():
        assert len(lanes) == len(set(lanes))


def test_failed_lane_indices_are_in_range():
    fail = DetectorFailure(1.0, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    for lanes in fail.failed.values():
        assert all(0 <= i < N_LANES for i in lanes)


def test_an_impossible_rate_is_rejected():
    with pytest.raises(ValueError):
        DetectorFailure(1.5, "zero")
    with pytest.raises(ValueError):
        DetectorFailure(-0.1, "zero")


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        DetectorFailure(0.25, "explode")


# ---------------------------------------------------------------------------
# What gets corrupted - and what must not
# ---------------------------------------------------------------------------
def test_the_phase_one_hot_and_min_green_flag_are_never_touched():
    """
    A signal always knows which phase it is showing. Corrupting that would
    model a different fault entirely.
    """
    fail = DetectorFailure(1.0, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    out = fail.corrupt(observations())
    for ts in IDS:
        assert np.allclose(out[ts][PHASE_SLICE], 1.0)


def test_stuck_at_zero_empties_both_readings_of_a_failed_lane():
    """Density and queue come from one sensor, so they fail together."""
    fail = DetectorFailure(0.25, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    out = fail.corrupt(observations(0.7))
    for ts in IDS:
        for lane in fail.failed[ts]:
            assert out[ts][DENSITY_SLICE.start + lane] == 0.0
            assert out[ts][QUEUE_SLICE.start + lane] == 0.0


def test_healthy_lanes_keep_their_true_readings():
    fail = DetectorFailure(0.25, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    out = fail.corrupt(observations(0.7))
    for ts in IDS:
        healthy = set(range(N_LANES)) - set(fail.failed[ts])
        for lane in healthy:
            assert out[ts][DENSITY_SLICE.start + lane] == pytest.approx(0.7)
            assert out[ts][QUEUE_SLICE.start + lane] == pytest.approx(0.7)


def test_corruption_does_not_mutate_the_environments_observation():
    """
    The metrics are read from the simulation, but an in-place edit would still
    hand the corrupted array to anything else holding a reference.
    """
    fail = DetectorFailure(1.0, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    original = observations(0.7)
    keep = {ts: obs.copy() for ts, obs in original.items()}
    fail.corrupt(original)
    for ts in IDS:
        assert np.allclose(original[ts], keep[ts])


def test_freeze_holds_the_previous_true_reading():
    fail = DetectorFailure(1.0, "freeze", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    fail.corrupt(observations(0.9))        # establishes the last good value
    out = fail.corrupt(observations(0.1))  # true value drops; frozen lanes hold
    for ts in IDS:
        for lane in fail.failed[ts]:
            assert out[ts][DENSITY_SLICE.start + lane] == pytest.approx(0.9)


def test_freeze_does_not_compound_its_own_corruption():
    """
    The remembered value must be the TRUE observation, not the one this wrapper
    just returned - otherwise "freeze" slowly becomes "stuck at whatever was
    first seen" and stops modelling the fault it claims to.
    """
    fail = DetectorFailure(1.0, "freeze", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    fail.corrupt(observations(0.9))
    fail.corrupt(observations(0.5))
    out = fail.corrupt(observations(0.1))
    for ts in IDS:
        for lane in fail.failed[ts]:
            # Holds the previous step's true 0.5, not the original 0.9.
            assert out[ts][DENSITY_SLICE.start + lane] == pytest.approx(0.5)


def test_freeze_passes_the_first_observation_through_unchanged():
    """With no previous reading there is nothing to hold."""
    fail = DetectorFailure(1.0, "freeze", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    out = fail.corrupt(observations(0.4))
    for ts in IDS:
        assert np.allclose(out[ts][DENSITY_SLICE], 0.4)


def test_the_two_modes_give_different_results():
    zero = DetectorFailure(0.5, "zero", scenario="s", seed=1)
    freeze = DetectorFailure(0.5, "freeze", scenario="s", seed=1)
    for p in (zero, freeze):
        p.on_episode_start(FakeEnv())
        p.corrupt(observations(0.9))
    a = zero.corrupt(observations(0.3))
    b = freeze.corrupt(observations(0.3))
    assert not np.allclose(a["J1"], b["J1"])


# ---------------------------------------------------------------------------
# The null perturbation
# ---------------------------------------------------------------------------
def test_zero_rate_returns_the_observations_untouched():
    fail = DetectorFailure(0.0, "zero", scenario="s", seed=1)
    fail.on_episode_start(FakeEnv())
    original = observations(0.6)
    assert fail.corrupt(original) is original


def test_make_perturbation_returns_the_null_for_a_zero_rate():
    assert isinstance(make_perturbation(0.0, "zero", "s", 1), NoPerturbation)


def test_make_perturbation_returns_a_real_failure_otherwise():
    p = make_perturbation(0.25, "freeze", "calibrated", 100)
    assert isinstance(p, DetectorFailure)
    assert p.rate == 0.25 and p.mode == "freeze"


def test_the_null_perturbation_reports_itself_as_clean():
    null = NoPerturbation()
    assert null.rate == 0.0 and null.mode == "none"
    assert null.corrupt(observations()) is not None


def test_every_declared_mode_is_usable():
    for mode in MODES:
        p = DetectorFailure(0.25, mode, scenario="s", seed=1)
        p.on_episode_start(FakeEnv())
        assert p.corrupt(observations()) is not None
