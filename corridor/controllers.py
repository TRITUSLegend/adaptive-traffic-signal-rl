"""
Corridor controllers: the baselines, the heuristic, and the learned-policy wrappers.

Every controller here exposes the same three-member interface the corridor's
episode runner drives, so a stopwatch, SUMO's own gap-out logic, a heuristic
and three neural networks are all measured by identical code:

    uses_env_control   False means SUMO runs its own program and `act` is ignored
    on_episode_start(env)
    act(observations) -> {ts_id: phase index}

`act` takes and returns DICTS, and must tolerate a dict that does not contain
every junction: sumo-rl only asks for a decision from signals whose
`time_to_act` is true, so agents act asynchronously (see corridor/env.py).

THE ACTUATED BASELINE IS NOT THE GREEDY HEURISTIC
-------------------------------------------------
`greedy-queue` picks whichever phase currently has the longest queue, with
global knowledge, every 5 s.  A real actuated controller does something
structurally different: it runs a fixed phase SEQUENCE and decides only when to
terminate the current green, from detector gap-out, bounded by a minimum and
maximum duration.

Relabelling the heuristic "actuated" would be exactly the sort of quiet
overstatement this project has twice had to remove (defect F4; the inert
`fixed-time-tuned+safe` wrapper).  So the actuated baseline is SUMO's own
`tlLogic type="actuated"`, compiled into a separate network file by
`build_corridor.py`, and the heuristic keeps its honest name.

A consequence worth stating before anyone reads a table: because SUMO runs the
actuated program itself, `ActuatedController.act` is never consulted, and the
safety wrapper is **inert on it by construction** - the same null Addendum E
documents for `fixed-time-tuned+safe`.  Its bounds are instead guaranteed by
the program's own `minDur`/`maxDur`, which `build_corridor.py` pins to the
project's MIN_GREEN and MAX_GREEN.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.simulation import DELTA_TIME  # noqa: E402
from eval.simulation import SafetyConstrainedController as _SingleSafety  # noqa: E402


class CorridorController:
    """Base class defining the interface `run_corridor_episode` drives."""

    name = "controller"
    uses_env_control = True

    def __init__(self, env=None):
        self.env = env

    def on_episode_start(self, env) -> None:
        self.env = env

    def act(self, observations: dict) -> dict:
        raise NotImplementedError


class FixedTimeCorridorController(CorridorController):
    """
    The baseline: every junction cycles its phases on a stopwatch.

    SUMO runs the static plan compiled into the network, so this class decides
    nothing.  It exists so the fixed-time signal is driven through the same
    episode runner as everything else, which is what guarantees it is measured
    identically rather than merely similarly.
    """

    name = "fixed-time"
    uses_env_control = False

    def act(self, observations: dict) -> dict:
        return {}  # ignored; SUMO is driving


class ActuatedCorridorController(CorridorController):
    """
    The second required baseline: SUMO's gap-based actuated control.

    Green is extended while vehicles keep crossing the detector within
    `max-gap` seconds of each other, and terminates on gap-out or on reaching
    `maxDur`.  That is the conventional meaning of "actuated" in this
    literature and it is implemented by SUMO rather than by us, which is the
    point: it is a standard, independently-specified baseline.

    Like fixed-time, SUMO drives it, so `act` is never consulted.  The
    distinction from fixed-time lives entirely in which network file the
    episode runner loads.
    """

    name = "actuated"
    uses_env_control = False

    def act(self, observations: dict) -> dict:
        return {}  # ignored; SUMO is driving


class GreedyQueueCorridorController(CorridorController):
    """
    A traffic-responsive reference controller that does no learning.

    Each junction independently serves whichever of its phases currently has
    the most stopped vehicles waiting.  Included because "RL beats a signal
    that ignores traffic" is a weak claim - at the single intersection this
    heuristic beat the DQN on 3 of 6 scenarios, and it is the honest
    competition here too.
    """

    name = "greedy-queue"
    uses_env_control = True

    def __init__(self, env=None):
        super().__init__(env)
        self._slots = {}

    def on_episode_start(self, env) -> None:
        self.env = env
        # Rebuilt per episode: the mapping is read from the live simulation.
        self._slots = {}

    def _phase_lane_slots(self, ts_id: str) -> list:
        """
        For each phase of one junction, the indices into its queue vector that
        the phase would release.

        SUMO knows which incoming lane each link index belongs to, so the light
        string is translated into lanes and then into positions in the vector
        `get_lanes_queue()` returns.
        """
        signal = self.env.traffic_signals[ts_id]
        controlled = self.env.sumo.trafficlight.getControlledLinks(ts_id)
        lane_of_link = {}
        for index, group in enumerate(controlled):
            if group:
                lane_of_link[index] = group[0][0]  # (inLane, outLane, via)

        slots = []
        for phase in signal.green_phases:
            released = set()
            for index, colour in enumerate(phase.state):
                if colour in "Gg" and index in lane_of_link:
                    lane = lane_of_link[index]
                    if lane in signal.lanes:
                        released.add(signal.lanes.index(lane))
            slots.append(sorted(released))
        return slots

    def act(self, observations: dict) -> dict:
        """
        Serve the longest queue, reading the queues FROM THE OBSERVATION.

        The obvious implementation asks the live simulation instead
        (`signal.get_lanes_queue()`), and the first version here did. That is
        wrong for one decisive reason: it bypasses the detector-failure
        wrapper, which corrupts the observation and nothing else. Measured on
        the `calibrated` hour, this controller returned bit-identical results
        at 0%, 12.5%, 25% and 50% of detectors failed - 31.837 s every time -
        because it was never reading the failed detectors at all.

        Published as-is, that table would have said the queue heuristic is
        perfectly robust to losing half its sensors. It is not; it simply had
        an oracle. This is the same shape as defect F4 in the single-
        intersection study, where a safety wrapper produced bit-identical
        traces and the identity was misread as compliance.

        The observation holds exactly what a detector would report: sumo-rl
        builds its queue block from `get_lanes_queue()` in `signal.lanes`
        order, which is the order `_phase_lane_slots` indexes. So under clean
        conditions this is numerically identical to the old version, and no
        clean number changes - it only stops the controller seeing through a
        failure it is supposed to suffer.
        """
        actions = {}
        for ts_id, observation in observations.items():
            if ts_id not in self._slots:
                self._slots[ts_id] = self._phase_lane_slots(ts_id)
            queues = self._queues_from(observation, ts_id)
            scores = [sum(queues[i] for i in slots)
                      for slots in self._slots[ts_id]]
            actions[ts_id] = int(max(range(len(scores)), key=scores.__getitem__))
        return actions

    def _queues_from(self, observation, ts_id: str) -> list:
        """
        Pull the per-lane queue block out of a sumo-rl observation.

        Layout: [phase one-hot][min-green flag][densities][queues], so the
        queues are the final `n_lanes` values.
        """
        signal = self.env.traffic_signals[ts_id]
        n_lanes = len(signal.lanes)
        return list(observation[-n_lanes:])


class IndependentPolicyController(CorridorController):
    """
    Wraps trained per-junction policies so they look like one controller.

    `policies` maps a traffic-signal id to anything with a stable-baselines3
    `.predict(obs, deterministic=True)`.  A single policy object may be shared
    by every junction - that is how a single-intersection model is evaluated on
    the corridor in the transfer experiment.
    """

    uses_env_control = True

    def __init__(self, policies, name: str, env=None):
        super().__init__(env)
        self._policies = policies
        self.name = name

    def _policy_for(self, ts_id: str):
        if isinstance(self._policies, dict):
            # Fall back to any single policy, which is the shared-policy case.
            if ts_id in self._policies:
                return self._policies[ts_id]
            if len(self._policies) == 1:
                return next(iter(self._policies.values()))
            raise KeyError("no policy for junction %r" % ts_id)
        return self._policies

    def act(self, observations: dict) -> dict:
        actions = {}
        for ts_id, obs in observations.items():
            # deterministic=True drops the exploration noise used in training:
            # at evaluation we want the policy's best guess, not a sample.
            action, _ = self._policy_for(ts_id).predict(obs, deterministic=True)
            actions[ts_id] = int(action)
        return actions


class SafetyConstrainedCorridorController(CorridorController):
    """
    Applies the maximum-green and anti-starvation bounds at every junction.

    This is the corridor version of `eval.simulation.SafetyConstrainedController`
    and it inherits that class's hard-won correctness condition, which is worth
    restating because getting it wrong produced a wrapper that enforced nothing
    while reporting compliance (defect F4):

        A phase counts as SERVED when the signal is actually showing it green,
        never when the wrapped controller merely ASKED for it.

    sumo-rl refuses a phase change while the current green is younger than
    `yellow + min_green`, so requests are frequently not granted.  The clocks
    below are therefore advanced from `signal.green_phase` - what the signal is
    doing - and never from the action returned by the inner controller.

    Bounds are taken from the single-intersection class so the two studies
    cannot drift apart.
    """

    MAX_GREEN = _SingleSafety.MAX_GREEN
    MAX_STARVATION = _SingleSafety.MAX_STARVATION

    def __init__(self, inner: CorridorController, env=None,
                 max_green: int = None, max_starvation: int = None):
        super().__init__(env)
        self.inner = inner
        self.max_green = self.MAX_GREEN if max_green is None else max_green
        self.max_starvation = (self.MAX_STARVATION if max_starvation is None
                               else max_starvation)
        self.name = inner.name + "+safe"
        self.uses_env_control = inner.uses_env_control
        self._green_held = {}
        self._since_served = {}
        self._current = {}
        self._observed_once = False
        self.max_green_overrides = 0
        self.starvation_overrides = 0

    def on_episode_start(self, env) -> None:
        self.env = env
        self.inner.on_episode_start(env)
        self._green_held = {}
        self._since_served = {}
        self._current = {}
        self._observed_once = False
        for ts_id in env.ts_ids:
            signal = env.traffic_signals[ts_id]
            # `num_green_phases`, not `len(green_phases)`: sumo-rl only builds
            # the list when it is driving the signal, so reading the list
            # raised AttributeError when wrapping a SUMO-driven plan (F1).
            self._since_served[ts_id] = {
                i: 0 for i in range(signal.num_green_phases)}
            self._green_held[ts_id] = 0
            self._current[ts_id] = None

    def _observe(self) -> None:
        """Advance every junction's clocks from what its signal is DOING."""
        for ts_id in self.env.ts_ids:
            signal = self.env.traffic_signals[ts_id]
            if self._observed_once:
                for phase in self._since_served[ts_id]:
                    self._since_served[ts_id][phase] += DELTA_TIME

            if getattr(signal, "is_yellow", False):
                # Amber is not a service: no phase has right of way, so it
                # neither extends the current green nor resets anything.
                self._green_held[ts_id] = 0
                self._current[ts_id] = None
                continue

            current = int(signal.green_phase)
            self._since_served[ts_id][current] = 0
            if current == self._current[ts_id]:
                self._green_held[ts_id] += DELTA_TIME
            else:
                self._green_held[ts_id] = DELTA_TIME
                self._current[ts_id] = current
        self._observed_once = True

    def act(self, observations: dict) -> dict:
        if not self.uses_env_control:
            # SUMO runs its own program; `green_phase` is not maintained, so
            # the wrapper can neither observe the signal nor change it. Pass
            # through and leave the counters at zero rather than keeping a
            # clock that would look like evidence and mean nothing.
            return self.inner.act(observations)

        self._observe()
        requested = self.inner.act(observations)

        chosen = {}
        for ts_id, want in requested.items():
            waits = self._since_served[ts_id]
            # Starvation first: it is the stronger guarantee, and a phase that
            # has waited too long should be served even if the current one has
            # not yet reached its maximum green.
            starved = [p for p, waited in waits.items()
                       if waited >= self.max_starvation and p != want]
            if starved:
                chosen[ts_id] = max(starved, key=lambda p: waits[p])
                self.starvation_overrides += 1
            elif (want == self._current[ts_id]
                  and self._green_held[ts_id] >= self.max_green):
                others = [p for p in waits if p != self._current[ts_id]]
                chosen[ts_id] = max(others, key=lambda p: waits[p])
                self.max_green_overrides += 1
            else:
                chosen[ts_id] = want
        # No clock update here on purpose: the clocks move in `_observe`, from
        # what the signal did. Updating them from a request is the F4 mistake.
        return chosen


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def fixed_time_factory():
    return lambda env: FixedTimeCorridorController(env)


def actuated_factory():
    return lambda env: ActuatedCorridorController(env)


def greedy_factory():
    return lambda env: GreedyQueueCorridorController(env)


def policy_factory(policies, name: str):
    return lambda env: IndependentPolicyController(policies, name, env)


def safety_wrapped_factory(inner_factory, max_green: int = None,
                           max_starvation: int = None):
    def build(env):
        return SafetyConstrainedCorridorController(
            inner_factory(env), env, max_green=max_green,
            max_starvation=max_starvation)
    return build
