"""
Tests for the max-green / anti-starvation wrapper (eval/simulation.py).

WHY THESE TESTS EXIST
---------------------
REPORT.md's limitations section says plainly that nothing in the project is
safety-validated: a 10 s minimum green and a 4 s amber were the only
constraints, and neither stops a learned policy from holding one phase forever
or from alternating between two while a third is never served.

Those failure modes are invisible in the headline metric. A starved approach
holds few enough vehicles that its delay is diluted by the ones being served
well, so mean waiting time can look fine while one movement is never released.
That makes the constraint exactly the kind of thing that has to be tested
directly rather than inferred from the numbers it produces.

The wrapper is driven here against a stub controller and a stub environment,
so every test is deterministic and runs without SUMO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.simulation import (  # noqa: E402
    DELTA_TIME,
    Controller,
    SafetyConstrainedController,
    safety_wrapped_factory,
)

N_PHASES = 4


class StubSignal:
    """
    Stands in for sumo-rl's TrafficSignal.

    It publishes `num_green_phases` and NOT `green_phases`, deliberately: that
    is exactly the shape sumo-rl presents when SUMO drives the signal, and the
    wrapper used to read the missing attribute and crash on any fixed-time
    controller. Keeping the stub minimal is what makes that a test rather than
    a coincidence.
    """

    num_green_phases = N_PHASES


class StubEnv:
    ts_ids = ["C"]
    traffic_signals = {"C": StubSignal()}


class Fixed(Controller):
    """Always asks for the same phase - the pathological 'never let go' case."""

    name = "stub-fixed"

    def __init__(self, phase=0, env=None):
        super().__init__(env)
        self.phase = phase

    def act(self, observation):
        return self.phase


class Alternating(Controller):
    """
    Ping-pongs between two phases, never serving the other two.

    A maximum green alone does NOT catch this: neither phase is ever held too
    long. Only the starvation bound does.
    """

    name = "stub-alternating"

    def __init__(self, a=0, b=2, env=None):
        super().__init__(env)
        self.pair = (a, b)
        self.i = 0

    def act(self, observation):
        self.i += 1
        return self.pair[self.i % 2]


class Scripted(Controller):
    name = "stub-scripted"

    def __init__(self, script, env=None):
        super().__init__(env)
        self.script = list(script)
        self.i = -1

    def act(self, observation):
        self.i += 1
        return self.script[min(self.i, len(self.script) - 1)]


def drive(controller, steps):
    """Run the wrapper for `steps` decisions and return the phases served."""
    env = StubEnv()
    controller.on_episode_start(env)
    return [controller.act(None) for _ in range(steps)]


# ---------------------------------------------------------------------------
# Maximum green
# ---------------------------------------------------------------------------
def test_a_phase_is_never_held_past_the_maximum_green():
    """THE CAP. A controller that never yields is made to yield."""
    wrapper = SafetyConstrainedController(Fixed(0), max_green=30,
                                          max_starvation=1000)
    served = drive(wrapper, 40)

    run, longest = 0, 0
    previous = None
    for phase in served:
        run = run + 1 if phase == previous else 1
        longest = max(longest, run)
        previous = phase
    assert longest * DELTA_TIME <= 30


def test_the_override_is_counted():
    wrapper = SafetyConstrainedController(Fixed(0), max_green=30,
                                          max_starvation=1000)
    drive(wrapper, 40)
    assert wrapper.max_green_overrides > 0


def test_a_controller_that_yields_on_its_own_is_never_overridden():
    """
    The wrapper must be inert for a controller that already behaves. If it
    intervened anyway it would be changing the thing being measured.
    """
    wrapper = SafetyConstrainedController(
        Scripted([0, 1, 2, 3] * 20), max_green=30, max_starvation=1000)
    served = drive(wrapper, 60)
    assert wrapper.max_green_overrides == 0
    assert wrapper.starvation_overrides == 0
    assert served == [0, 1, 2, 3] * 15


def test_the_cap_hands_over_to_the_longest_waiting_phase():
    """Handing over to the most-starved phase is what shortens the worst wait."""
    wrapper = SafetyConstrainedController(Fixed(0), max_green=10,
                                          max_starvation=1000)
    env = StubEnv()
    wrapper.on_episode_start(env)
    wrapper.act(None)          # t=5   phase 0
    wrapper.act(None)          # t=10  phase 0, now at the cap
    third = wrapper.act(None)  # must move on
    assert third != 0
    # Phases 1, 2 and 3 have all waited the same, so the tie is broken
    # deterministically rather than arbitrarily.
    assert third in (1, 2, 3)


# ---------------------------------------------------------------------------
# Starvation
# ---------------------------------------------------------------------------
def test_alternating_between_two_phases_is_caught():
    """
    THE CASE A MAXIMUM GREEN MISSES. Phases 1 and 3 are never requested, and
    neither served phase is ever held long enough to trip the cap.
    """
    inner = Alternating(0, 2)
    wrapper = SafetyConstrainedController(inner, max_green=60,
                                          max_starvation=100)
    served = drive(wrapper, 80)
    assert set(served) == {0, 1, 2, 3}, "a phase was never served"
    assert wrapper.starvation_overrides > 0


@pytest.mark.parametrize("bound", [50, 100, 200])
def test_no_phase_ever_waits_longer_than_the_bound(bound):
    """The guarantee itself, checked against the worst gap actually observed."""
    wrapper = SafetyConstrainedController(Fixed(0), max_green=60,
                                          max_starvation=bound)
    served = drive(wrapper, 200)

    last_seen = {p: -1 for p in range(N_PHASES)}
    worst = 0
    for step, phase in enumerate(served):
        last_seen[phase] = step
        for other, seen in last_seen.items():
            if other != phase:
                worst = max(worst, (step - seen) * DELTA_TIME)
    # Only one phase can hold green at a time, so when several cross the bound
    # together they must be worked through one decision apart. The guaranteed
    # ceiling is the bound plus (n_phases - 1) steps - 15 s here - not the
    # bound exactly. This test originally asserted the tighter bound and
    # failed, which is how that property was pinned down.
    assert worst <= bound + (N_PHASES - 1) * DELTA_TIME


def test_every_phase_is_served_within_the_bound_from_the_start():
    wrapper = SafetyConstrainedController(Fixed(0), max_green=60,
                                          max_starvation=100)
    served = drive(wrapper, 40)
    first_service = {}
    for step, phase in enumerate(served):
        first_service.setdefault(phase, step * DELTA_TIME)
    assert set(first_service) == set(range(N_PHASES))
    assert max(first_service.values()) <= 100 + DELTA_TIME


def test_starvation_takes_priority_over_the_green_cap():
    """
    Both constraints can fire on the same step. The starvation bound is the
    stronger guarantee, so it decides.
    """
    wrapper = SafetyConstrainedController(Fixed(0), max_green=10,
                                          max_starvation=15)
    drive(wrapper, 30)
    assert wrapper.starvation_overrides > 0


# ---------------------------------------------------------------------------
# The wrapper must not change what it is wrapping
# ---------------------------------------------------------------------------
def test_the_wrapper_reports_a_distinct_name():
    wrapper = SafetyConstrainedController(Fixed(0))
    assert wrapper.name == "stub-fixed+safe"


def test_the_wrapper_inherits_who_drives_the_signal():
    """
    A fixed-time plan is run by SUMO, not by us. Wrapping must not silently
    take control of the signal away from it.
    """
    inner = Fixed(0)
    inner.uses_env_control = False
    assert SafetyConstrainedController(inner).uses_env_control is False

    inner.uses_env_control = True
    assert SafetyConstrainedController(inner).uses_env_control is True


def test_the_wrapper_starts_on_a_signal_sumo_is_driving():
    """
    THE REGRESSION. sumo-rl builds its `green_phases` list only when it is
    driving the signal itself. With a fixed-time plan SUMO runs the program,
    the attribute is absent, and `on_episode_start` used to raise
    AttributeError before the episode had taken a step - so `dqn+safe` ran
    fine and `fixed-time-tuned+safe` could not be evaluated at all.

    The phase count is read from `num_green_phases`, which is published in
    both modes. Asserting the attribute is absent is the point of the test:
    without that, a stub carrying both would pass either way.
    """
    signal = StubEnv.traffic_signals["C"]
    assert not hasattr(signal, "green_phases")

    wrapper = SafetyConstrainedController(Fixed(0))
    wrapper.on_episode_start(StubEnv())

    assert sorted(wrapper._since_served) == list(range(N_PHASES))


def test_the_inner_controller_gets_the_episode_start_hook():
    seen = []

    class Recording(Fixed):
        def on_episode_start(self, env):
            seen.append(env)

    wrapper = SafetyConstrainedController(Recording(0))
    env = StubEnv()
    wrapper.on_episode_start(env)
    assert seen == [env]


def test_counters_reset_between_episodes():
    wrapper = SafetyConstrainedController(Fixed(0), max_green=10,
                                          max_starvation=1000)
    drive(wrapper, 20)
    wrapper.on_episode_start(StubEnv())
    assert wrapper._green_held == 0
    assert all(v == 0 for v in wrapper._since_served.values())


def test_the_factory_builds_a_wrapped_controller():
    factory = safety_wrapped_factory(lambda env: Fixed(0, env), max_green=30)
    controller = factory(StubEnv())
    assert isinstance(controller, SafetyConstrainedController)
    assert controller.max_green == 30
    assert controller.max_starvation == SafetyConstrainedController.MAX_STARVATION


def test_the_factory_tolerates_the_none_probe():
    """
    `run_episode` builds a controller with env=None purely to read
    uses_env_control before the environment exists. The wrapper must survive
    that probe.
    """
    factory = safety_wrapped_factory(lambda env: Fixed(0, env))
    assert factory(None).uses_env_control is True


def test_defaults_are_looser_than_any_sensible_fixed_plan():
    """
    The untuned plan gives a through phase 32 s and serves every phase inside a
    104 s cycle, so the defaults must not bind on it - otherwise the wrapper
    would be altering the baseline rather than constraining bad behaviour.
    """
    assert SafetyConstrainedController.MAX_GREEN > 32
    assert SafetyConstrainedController.MAX_STARVATION > 104
