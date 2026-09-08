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

WHAT THE STUB HAS TO MODEL, AND WHY
-----------------------------------
The first version of these tests used a signal stub that simply did what it was
told. That let a wrapper pass every test while enforcing nothing, because the
real environment does *not* simply do what it is told: sumo-rl refuses a phase
change until the current green has run for `yellow + min_green`. The wrapper
counted a phase as served the moment it was asked for, so a controller that
asked for a phase it never received kept resetting that phase's clock, and the
starvation bound never fired. On one measured `balanced` hour the phase the
greedy heuristic asked for was not the phase then green on 573 of 720
decisions; the wrapper's own worst clock read 175 s, under its 200 s bound,
while the junction left a phase unserved for 245 s.

So `FakeSignal` refuses changes the way sumo-rl does, and `drive` returns what
the signal actually *served* rather than what was asked for. Every guarantee
below is asserted against the served sequence. The tests still run without
SUMO and are still deterministic.
"""

from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.simulation import (  # noqa: E402
    DELTA_TIME,
    MIN_GREEN,
    YELLOW_TIME,
    Controller,
    SafetyConstrainedController,
    safety_wrapped_factory,
)

N_PHASES = 4

# sumo-rl will not change phase while the current green is younger than this.
REFUSAL_WINDOW = YELLOW_TIME + MIN_GREEN            # 14 s

# The same window rounded up to a whole decision, which is the granularity the
# wrapper can actually act at: a hand-over costs at most this much.
HANDOVER = -(-REFUSAL_WINDOW // DELTA_TIME) * DELTA_TIME   # 15 s

# Worst case for the starvation guarantee. Only one phase holds green at a
# time, so if every other phase crosses the bound together they are worked
# through one hand-over apart.
def ceiling(bound: int) -> int:
    return bound + (N_PHASES - 1) * HANDOVER


class FakeSignal:
    """
    Stands in for sumo-rl's TrafficSignal, including the part that broke the
    first wrapper.

    Publishes `num_green_phases`, `green_phase` and `is_yellow`, and
    deliberately **not** `green_phases` - sumo-rl builds that list only when it
    is driving the signal itself, and the wrapper once read it and crashed on
    every fixed-time controller.

    `apply` is the refusal rule: a requested phase is adopted only if the
    current green has already run for `refuse_within`. Otherwise the request is
    silently dropped and the current phase carries on, which is what sumo-rl
    does and what a wrapper has to cope with.
    """

    def __init__(self, n_phases: int = N_PHASES,
                 refuse_within: int = REFUSAL_WINDOW):
        self.num_green_phases = n_phases
        self.green_phase = 0
        self.is_yellow = False
        self.refuse_within = refuse_within
        self.held = 0        # seconds the current green has been showing
        self.history = []    # the phase actually green over each interval

    def apply(self, requested: int) -> None:
        if requested != self.green_phase and self.held >= self.refuse_within:
            self.green_phase = requested
            self.held = DELTA_TIME
        else:
            self.held += DELTA_TIME
        self.history.append(self.green_phase)


class FakeEnv:
    def __init__(self, signal: FakeSignal = None):
        self.ts_ids = ["C"]
        self.signal = signal if signal is not None else FakeSignal()
        self.traffic_signals = {"C": self.signal}


Run = namedtuple("Run", "requested served signal")


def drive(controller, steps: int, signal: FakeSignal = None) -> Run:
    """
    Run the wrapper for `steps` decisions against a signal that behaves like
    sumo-rl's.

    Returns both what was asked for and what was served. The distinction is the
    whole point: the first wrapper was correct about its own requests and wrong
    about the junction, and a harness that only recorded requests could not
    tell the difference.
    """
    env = FakeEnv(signal)
    controller.on_episode_start(env)
    requested = []
    for _ in range(steps):
        action = controller.act(None)
        requested.append(action)
        env.signal.apply(action)
    return Run(requested, env.signal.history, env.signal)


def longest_run(served: list) -> int:
    """Longest stretch of consecutive intervals on one phase, in seconds."""
    longest, run, previous = 0, 0, None
    for phase in served:
        run = run + 1 if phase == previous else 1
        longest = max(longest, run)
        previous = phase
    return longest * DELTA_TIME


def worst_unserved_gap(served: list) -> int:
    """
    Longest any phase went without green, in seconds, over the whole run.

    Measured from the start and to the end, not only between services: a phase
    served once at the beginning and never again is starved for the rest of the
    run, and a version that looked only at the intervals between services would
    score that as zero.
    """
    end = len(served) * DELTA_TIME
    worst = 0
    for phase in range(N_PHASES):
        times = [(i + 1) * DELTA_TIME for i, p in enumerate(served) if p == phase]
        boundaries = [0] + times + [end]
        worst = max(worst,
                    max(b - a for a, b in zip(boundaries, boundaries[1:])))
    return worst


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
    Ping-pongs between two phases, never asking for the other two.

    A maximum green alone does NOT catch this: neither served phase is ever
    held too long. Only the starvation bound does.
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


class Teasing(Controller):
    """
    Asks for phase 3 only at the moments the signal will refuse it, and for
    something else the rest of the time.

    Phase 3 is therefore requested constantly and served never - the exact
    shape that defeated the first wrapper, which treated "asked for" as
    "served" and so believed phase 3 was being looked after.
    """

    name = "stub-teasing"

    def __init__(self, signal, env=None):
        super().__init__(env)
        self.signal = signal

    def act(self, observation):
        if self.signal.held < self.signal.refuse_within:
            return 3
        return 1 if self.signal.green_phase == 0 else 0


# ---------------------------------------------------------------------------
# The clocks must follow the signal, not the requests
# ---------------------------------------------------------------------------
def test_a_refused_request_does_not_count_as_a_service():
    """
    THE REGRESSION, in its smallest form. The signal never grants a change, so
    phase 3 is asked for on every decision and served on none. Its clock must
    keep climbing; the first wrapper reset it to zero on every request.
    """
    signal = FakeSignal(refuse_within=10_000)
    wrapper = SafetyConstrainedController(Fixed(3), max_starvation=10_000)
    run = drive(wrapper, 10, signal)

    assert run.requested == [3] * 10
    assert set(run.served) == {0}, "the signal cannot have moved"
    assert wrapper._since_served[3] == 9 * DELTA_TIME
    assert wrapper._since_served[0] == 0


def test_a_phase_asked_for_but_never_served_still_trips_the_bound():
    """
    THE REGRESSION, end to end. Under the first wrapper phase 3 looked freshly
    served on two decisions in every three and the bound never fired, while the
    junction never showed it green at all.
    """
    signal = FakeSignal()
    wrapper = SafetyConstrainedController(Teasing(signal), max_green=1000,
                                          max_starvation=100)
    run = drive(wrapper, 120, signal)

    assert 3 in run.served, "the starved phase was never actually served"
    assert wrapper.starvation_overrides > 0
    assert worst_unserved_gap(run.served) <= ceiling(100)


def test_amber_is_not_counted_as_serving_anybody():
    """During amber no phase has right of way, so no clock may reset."""
    signal = FakeSignal()
    wrapper = SafetyConstrainedController(Fixed(0), max_starvation=10_000)
    wrapper.on_episode_start(FakeEnv(signal))
    wrapper.act(None)
    signal.is_yellow = True
    before = dict(wrapper._since_served)
    wrapper.act(None)

    assert all(wrapper._since_served[p] == before[p] + DELTA_TIME
               for p in before)
    assert wrapper._green_held == 0


# ---------------------------------------------------------------------------
# Maximum green
# ---------------------------------------------------------------------------
def test_a_phase_is_never_held_past_the_maximum_green():
    """THE CAP. A controller that never yields is made to yield."""
    wrapper = SafetyConstrainedController(Fixed(0), max_green=30,
                                          max_starvation=10_000)
    run = drive(wrapper, 40)
    assert longest_run(run.served) <= 30


def test_the_override_is_counted():
    wrapper = SafetyConstrainedController(Fixed(0), max_green=30,
                                          max_starvation=10_000)
    drive(wrapper, 40)
    assert wrapper.max_green_overrides > 0


def test_a_controller_that_yields_on_its_own_is_never_overridden():
    """
    The wrapper must be inert for a controller that already behaves. If it
    intervened anyway it would be changing the thing being measured.

    Asserted on the REQUESTS, because the signal's refusal window means a
    well-behaved controller's requests and the signal's greens legitimately
    differ - that is the environment's doing, not the wrapper's.
    """
    script = [0, 1, 2, 3] * 20
    wrapper = SafetyConstrainedController(Scripted(script), max_green=30,
                                          max_starvation=10_000)
    run = drive(wrapper, 60)

    assert wrapper.max_green_overrides == 0
    assert wrapper.starvation_overrides == 0
    assert run.requested == script[:60]
    assert set(run.served) == set(range(N_PHASES))


def test_the_cap_hands_over_to_the_longest_waiting_phase():
    """Handing over to the most-starved phase is what shortens the worst wait."""
    signal = FakeSignal()
    wrapper = SafetyConstrainedController(Fixed(0), max_green=10,
                                          max_starvation=10_000)
    wrapper.on_episode_start(FakeEnv(signal))

    signal.apply(wrapper.act(None))   # t=5   phase 0
    signal.apply(wrapper.act(None))   # t=10  phase 0, now at the cap
    third = wrapper.act(None)         # must ask for something else

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
    wrapper = SafetyConstrainedController(Alternating(0, 2), max_green=60,
                                          max_starvation=100)
    run = drive(wrapper, 120)

    assert set(run.served) == {0, 1, 2, 3}, "a phase was never served"
    assert wrapper.starvation_overrides > 0


@pytest.mark.parametrize("bound", [50, 100, 200])
def test_no_phase_ever_waits_longer_than_the_bound(bound):
    """
    THE GUARANTEE, checked against the gap the signal actually left.

    The ceiling is the bound plus one hand-over per other phase, not the bound
    exactly, and both parts of that slack are forced rather than sloppy: only
    one phase can hold green at a time, and sumo-rl will not hand over until
    the current green has run for yellow + min_green. On the real network the
    measured worst gap is 210-215 s against a 200 s bound, comfortably inside
    this ceiling, because phases rarely all starve at once.
    """
    wrapper = SafetyConstrainedController(Fixed(0), max_green=1000,
                                          max_starvation=bound)
    run = drive(wrapper, 300)
    assert worst_unserved_gap(run.served) <= ceiling(bound)


def test_every_phase_is_served_within_the_bound_from_the_start():
    wrapper = SafetyConstrainedController(Fixed(0), max_green=1000,
                                          max_starvation=100)
    run = drive(wrapper, 120)

    first_service = {}
    for step, phase in enumerate(run.served):
        first_service.setdefault(phase, (step + 1) * DELTA_TIME)

    assert set(first_service) == set(range(N_PHASES))
    assert max(first_service.values()) <= ceiling(100)


def test_starvation_takes_priority_over_the_green_cap():
    """
    Both constraints can fire on the same step. The starvation bound is the
    stronger guarantee, so it decides.
    """
    wrapper = SafetyConstrainedController(Fixed(0), max_green=10,
                                          max_starvation=15)
    drive(wrapper, 40)
    assert wrapper.starvation_overrides > 0


def test_a_tighter_bound_produces_a_smaller_gap():
    """
    The bound has to actually control the outcome, not merely be satisfied by
    it. A wrapper that ignored its argument would pass every test above.
    """
    gaps = {}
    for bound in (50, 200):
        wrapper = SafetyConstrainedController(Fixed(0), max_green=1000,
                                              max_starvation=bound)
        gaps[bound] = worst_unserved_gap(drive(wrapper, 300).served)
    assert gaps[50] < gaps[200]


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


def test_a_wrapper_on_a_fixed_plan_passes_everything_through():
    """
    THE INERT CASE, pinned. SUMO runs the plan, our `act` is never consulted,
    and sumo-rl does not maintain the phase index the wrapper would need. It
    must pass the action through and leave its counters at zero rather than
    keep a clock that would look like evidence. Addendum E reports this row as
    an inert wrapper on the strength of this behaviour.
    """
    inner = Fixed(0)
    inner.uses_env_control = False
    wrapper = SafetyConstrainedController(inner, max_green=10,
                                          max_starvation=15)
    run = drive(wrapper, 30)

    assert run.requested == [0] * 30
    assert wrapper.max_green_overrides == 0
    assert wrapper.starvation_overrides == 0


def test_the_wrapper_starts_on_a_signal_sumo_is_driving():
    """
    sumo-rl builds its `green_phases` list only when it is driving the signal
    itself. With a fixed-time plan SUMO runs the program, the attribute is
    absent, and `on_episode_start` used to raise AttributeError before the
    episode had taken a step - so `dqn+safe` ran fine and
    `fixed-time-tuned+safe` could not be evaluated at all.

    The phase count is read from `num_green_phases`, which is published in
    both modes. Asserting the attribute is absent is the point of the test:
    without that, a stub carrying both would pass either way.
    """
    env = FakeEnv()
    assert not hasattr(env.signal, "green_phases")

    wrapper = SafetyConstrainedController(Fixed(0))
    wrapper.on_episode_start(env)

    assert sorted(wrapper._since_served) == list(range(N_PHASES))


def test_the_inner_controller_gets_the_episode_start_hook():
    seen = []

    class Recording(Fixed):
        def on_episode_start(self, env):
            seen.append(env)

    wrapper = SafetyConstrainedController(Recording(0))
    env = FakeEnv()
    wrapper.on_episode_start(env)
    assert seen == [env]


def test_counters_reset_between_episodes():
    wrapper = SafetyConstrainedController(Fixed(0), max_green=10,
                                          max_starvation=10_000)
    drive(wrapper, 20)
    wrapper.on_episode_start(FakeEnv())

    assert wrapper._green_held == 0
    assert all(v == 0 for v in wrapper._since_served.values())


def test_no_time_is_charged_before_the_first_decision():
    """
    An episode's first decision happens at t=0, so nothing may have elapsed
    yet. Charging one interval there would quietly shorten every bound by
    5 seconds.
    """
    wrapper = SafetyConstrainedController(Fixed(0), max_starvation=10_000)
    wrapper.on_episode_start(FakeEnv())
    wrapper.act(None)
    assert max(wrapper._since_served.values()) == 0


def test_the_factory_builds_a_wrapped_controller():
    factory = safety_wrapped_factory(lambda env: Fixed(0, env), max_green=30)
    controller = factory(FakeEnv())
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
