"""
A controller that claims to read detectors must actually read them.

WHY THIS TEST EXISTS
--------------------
The detector-failure study corrupts the OBSERVATION and nothing else. That is
what makes it a sensing failure rather than a change in demand. But it also
means a controller that quietly bypasses the observation - by asking the live
simulation for its queues instead - is immune to the perturbation for reasons
that have nothing to do with robustness.

That happened. `GreedyQueueCorridorController.act` called
`signal.get_lanes_queue()` directly, and the sweep returned bit-identical
results at 0%, 12.5%, 25% and 50% of detectors failed: 31.837 s every time.
Published unchecked, the report would have claimed the queue heuristic is
perfectly robust to losing half its sensors. It had an oracle instead.

This is defect F4's shape exactly - bit-identical output mistaken for a
property - and F4's lesson was that identity must be interrogated, never
trusted. So the property is asserted directly: feed a controller two
observations that differ only in their queue block, and a detector-dependent
controller must be capable of answering differently.

THE BASELINES ARE THE CONTROL
-----------------------------
Fixed-time and actuated are driven by SUMO and never consult `act()`, so their
immunity IS structural and correct. They are asserted to be immune, so that the
test distinguishes "immune by construction" from "immune by accident" rather
than merely flagging every unchanged number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor.controllers import (  # noqa: E402
    ActuatedCorridorController,
    FixedTimeCorridorController,
    GreedyQueueCorridorController,
)
from corridor.perturbations import QUEUE_SLICE  # noqa: E402

N_LANES = 8
N_PHASES = 4


class FakePhase:
    def __init__(self, state):
        self.state = state


class FakeSignal:
    """
    Enough of a sumo-rl TrafficSignal for the heuristic's phase map.

    Four phases over eight lanes, in the corridor's own layout: lanes 0-1 are
    the north approach, 2-3 south, 4-5 east, 6-7 west, with lane 0 of each pair
    carrying through+right and lane 1 the protected left.
    """

    def __init__(self):
        self.lanes = ["L%d" % i for i in range(N_LANES)]
        self.num_green_phases = N_PHASES
        self.green_phases = [
            FakePhase("GGrrrrGGrrrr"),   # side through+right: lanes 0, 4
            FakePhase("rrGrrrrrGrrr"),   # side left:          lanes 1, 5
            FakePhase("rrrGGrrrrGGr"),   # arterial through:   lanes 2, 6
            FakePhase("rrrrrGrrrrrG"),   # arterial left:      lanes 3, 7
        ]


class FakeTraci:
    def getControlledLinks(self, ts_id):
        # One link per (lane, movement); the mapping mirrors the real network's
        # 12 controlled movements over 8 lanes.
        order = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7]
        return [[("L%d" % lane, "out", "via")] for lane in order]


class FakeEnv:
    def __init__(self):
        self.ts_ids = ["J1"]
        self.traffic_signals = {"J1": FakeSignal()}
        self.sumo = type("S", (), {"trafficlight": FakeTraci()})()


def observation(queues) -> np.ndarray:
    """[phase one-hot][min-green][densities][queues] for one junction."""
    obs = np.zeros(N_PHASES + 1 + 2 * N_LANES, dtype=np.float32)
    obs[0] = 1.0
    obs[N_PHASES] = 1.0
    obs[QUEUE_SLICE] = np.asarray(queues, dtype=np.float32)
    return obs


@pytest.fixture
def greedy():
    env = FakeEnv()
    controller = GreedyQueueCorridorController(env)
    controller.on_episode_start(env)
    return controller


# ---------------------------------------------------------------------------
# The heuristic must respond to what the observation says
# ---------------------------------------------------------------------------
def test_the_heuristic_serves_the_phase_with_the_longest_queue(greedy):
    """Lane 2 is busiest, and lane 2 is released by phase 2."""
    queues = [0.0] * N_LANES
    queues[2] = 0.9
    queues[6] = 0.9
    assert greedy.act({"J1": observation(queues)})["J1"] == 2


def test_moving_the_queue_moves_the_decision(greedy):
    """The property the whole study depends on: the observation is consulted."""
    a = [0.0] * N_LANES
    a[0] = a[4] = 0.9                     # phase 0's lanes
    b = [0.0] * N_LANES
    b[3] = b[7] = 0.9                     # phase 3's lanes
    assert greedy.act({"J1": observation(a)})["J1"] == 0
    assert greedy.act({"J1": observation(b)})["J1"] == 3


def test_the_heuristic_reads_the_observation_not_the_simulation(greedy):
    """
    The regression. The fake signal exposes no live queue source at all, so a
    controller that tried to read one would raise rather than quietly work.
    Two observations differing only in their queue block must be answerable.
    """
    busy_side = [0.0] * N_LANES
    busy_side[0] = busy_side[4] = 1.0
    busy_arterial = [0.0] * N_LANES
    busy_arterial[2] = busy_arterial[6] = 1.0
    assert (greedy.act({"J1": observation(busy_side)})["J1"]
            != greedy.act({"J1": observation(busy_arterial)})["J1"])


def test_a_blinded_detector_changes_what_the_heuristic_sees(greedy):
    """
    Stuck-at-zero on the busiest lanes must be able to change the decision.
    This is the exact scenario the sweep exercises; if it cannot change the
    answer, the sweep measures nothing.
    """
    true_queues = [0.0] * N_LANES
    true_queues[2] = true_queues[6] = 0.9      # arterial is genuinely busy
    true_queues[0] = true_queues[4] = 0.4
    honest = greedy.act({"J1": observation(true_queues)})["J1"]

    blinded = list(true_queues)
    blinded[2] = blinded[6] = 0.0              # its detectors died
    deceived = greedy.act({"J1": observation(blinded)})["J1"]

    assert honest == 2
    assert deceived != honest, (
        "blinding the busiest approach did not change the decision, so the "
        "controller is not reading its detectors")


def test_an_all_zero_observation_still_yields_a_legal_phase(greedy):
    """Total detector failure must not crash the controller."""
    action = greedy.act({"J1": observation([0.0] * N_LANES)})["J1"]
    assert 0 <= action < N_PHASES


def test_the_heuristic_answers_only_for_junctions_it_was_given(greedy):
    assert set(greedy.act({"J1": observation([0.1] * N_LANES)})) == {"J1"}


# ---------------------------------------------------------------------------
# The baselines are immune BY CONSTRUCTION, which is a different thing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("controller_class",
                         [FixedTimeCorridorController, ActuatedCorridorController])
def test_sumo_driven_baselines_do_not_consult_the_observation(controller_class):
    """
    Their `act` is never called by sumo-rl in fixed_ts mode, and it returns an
    empty action dict whatever it is handed. Their zero degradation under
    detector failure is therefore arithmetic, not evidence - and the report
    says so rather than presenting it as robustness.
    """
    controller = controller_class(None)
    assert controller.uses_env_control is False
    busy = observation([1.0] * N_LANES)
    empty = observation([0.0] * N_LANES)
    assert controller.act({"J1": busy}) == controller.act({"J1": empty}) == {}
